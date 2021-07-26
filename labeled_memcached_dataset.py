# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------

from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class McDataset(Dataset):
    def __init__(self, data_root, file_list, phase = 'train', transform=None):
        self.transform = transform
        self.root = os.path.join(data_root, phase)
        
        temp_label = json.load(open('./dataset/imagenet_class_index.json', 'r'))
        self.labels = {}
        for i in range(1000):
            self.labels[temp_label[str(i)][0]] = i
        self.A_paths = []
        self.A_labels = []
        with open(file_list, 'r') as f:
            temp_path = f.readlines()
        for path in temp_path:
            label = self.labels[path.split('/')[0]]
            self.A_paths.append(os.path.join(self.root, path.strip()))
            self.A_labels.append(label)

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        return A, A_label
