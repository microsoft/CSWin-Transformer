# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import numpy as np
import time

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from ..builder import BACKBONES

import torch.utils.checkpoint as checkpoint


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Not supported now, since we have cls_tokens now.....
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.H_sp_ = self.H_sp
        self.W_sp_ = self.W_sp

        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, C, H, W = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_rpe(self, x, func):
        B, C, H, W = x.shape
        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        rpe = func(x) ### B', C, H', W'
        rpe = rpe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, rpe

    def forward(self, temp):
        """
        x: B N C
        mask: B N N
        """
        B, _, C, H, W = temp.shape


        idx = self.idx
        if idx == -1:
            H_sp, W_sp = H, W
        elif idx == 0:
            H_sp, W_sp = H, self.split_size
        elif idx == 1:
            H_sp, W_sp = self.split_size, W
        else:
            print ("ERROR MODE in forward", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        ### padding for split window
        H_pad = (self.H_sp - H % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - W % self.W_sp) % self.W_sp
        top_pad = H_pad//2
        down_pad = H_pad - top_pad
        left_pad = W_pad//2
        right_pad = W_pad - left_pad
        H_ = H + H_pad
        W_ = W + W_pad

        qkv = F.pad(temp, (left_pad, right_pad, top_pad, down_pad)) ### B,3,C,H',W'
        qkv = qkv.permute(1, 0, 2, 3, 4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, rpe = self.get_rpe(v, self.get_v)

        ### Local attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v) + rpe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H_, W_) # B H_ W_ C
        x = x[:, top_pad:H+top_pad, left_pad:W+left_pad, :]
        x = x.reshape(B, -1, C)

        return x

class CSWinBlock(nn.Module):

    def __init__(self, dim, patches_resolution, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = patches_resolution
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.norm1 = norm_layer(dim)

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

        atten_mask_matrix = None

        self.register_buffer("atten_mask_matrix", atten_mask_matrix)
        self.H = None
        self.W = None

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        H = self.H
        W = self.W
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        temp = self.qkv(img).reshape(B, H, W, 3, C).permute(0, 3, 4, 1, 2)
        
        if self.branch_num == 2:
            x1 = self.attns[0](temp[:,:,:C//2,:,:])
            x2 = self.attns[1](temp[:,:,C//2:,:,:])
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            attened_x = self.attns[0](temp)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x, H, W):
        B, new_HW, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x, H, W

@BACKBONES.register_module()
class CSWin(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64, depth=[1,2,21,1], split_size = 7,
                 num_heads=[1,2,4,8], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        heads=num_heads
        self.use_chk = use_chk
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h = img_size//4, w = img_size//4),
            nn.LayerNorm(embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], patches_resolution=224//4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*(heads[1]//heads[0]))
        curr_dim = curr_dim*(heads[1]//heads[0])
        self.norm2 = nn.LayerNorm(curr_dim)
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], patches_resolution=224//8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer)
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*(heads[2]//heads[1]))
        curr_dim = curr_dim*(heads[2]//heads[1])
        self.norm3 = nn.LayerNorm(curr_dim)
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], patches_resolution=224//16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*(heads[3]//heads[2]))
        curr_dim = curr_dim*(heads[3]//heads[2])
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], patches_resolution=224//32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True)
            for i in range(depth[-1])])
       
        self.norm4 = norm_layer(curr_dim)


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def save_out(self, x, norm, H, W):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stage1_conv_embed[0](x) ### B, C, H, W
        B, C, H, W = x.size()
        x = x.reshape(B, C, -1).transpose(-1,-2).contiguous()
        x = self.stage1_conv_embed[2](x)

        out = []
        for blk in self.stage1:
            blk.H = H
            blk.W = W
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        out.append(self.save_out(x, self.norm1, H, W))

        for pre, blocks, norm in zip([self.merge1, self.merge2, self.merge3], 
                                     [self.stage2, self.stage3, self.stage4],
                                     [self.norm2 , self.norm3 , self.norm4 ]):

            x, H, W = pre(x, H, W)
            for blk in blocks:
                blk.H = H
                blk.W = W
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            out.append(self.save_out(x, norm, H, W))

        return tuple(out)

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


