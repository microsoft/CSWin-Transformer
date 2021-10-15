# ADE20k Semantic segmentation with CSWin


## Results and Models

| Backbone | Method | pretrain | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | config | model | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |  :---: | :---: | :---: |
| CSWin-T | UPerNet | ImageNet-1K | 512x512 | 160K | 49.3 | 50.7 | 60M | 959G | [`config`](configs/cswin/upernet_cswin_tiny.py) | [model]() | [log]() |
| CSWin-S | UperNet | ImageNet-1K | 512x512 | 160K | 50.4 | 51.5 | 65M | 1027G |  [`config`](configs/cswin/upernet_cswin_small.py) |[model]() | [log]() |
| CSWin-B | UperNet | ImageNet-1K | 512x512 | 160K | 51.1 | 52.2 | 109M | 1222G | [`config`](configs/cswin/upernet_cswin_base.py) |[model]() | [log]() |


## Getting started 

1. Install the [Swin_Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) repository and some required packages.

```bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
bash install_req.sh
```

2. Move the CSWin configs and backbone file to the corresponding folder.

```bash
cp -r configs/cswin <MMSEG_PATH>/configs/
cp config/_base/upernet_cswin.py <MMSEG_PATH>/config/_base_/models
cp backbone/cswin_transformer.py <MMSEG_PATH>/mmseg/models/backbones/
cp mmcv_custom/checkpoint.py <MMSEG_PATH>/mmcv_custom/
```

3. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

4. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.

## Fine-tuning

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS> --options model.pretrained=<PRETRAIN_MODEL_PATH>
```

For example, using a CSWin-T backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/cswin/upernet_cswin_tiny.py 8 \
    --options model.pretrained=<PRETRAIN_MODEL_PATH>
```

pretrained models could be found at [main page](https://github.com/microsoft/CSWin-Transformer).

More config files can be found at [`configs/cswin`](configs/cswin).


## Evaluation

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```

For example, evaluate a CSWin-T backbone with UperNet:
```bash
bash tools/dist_test.sh configs/cswin/upernet_cswin_tiny.py \ 
    <CHECKPOINT_PATH> 8 --eval mIoU
```


---

## Acknowledgment 

This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository.
