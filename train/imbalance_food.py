import os
import numpy as np
import json
import math
import PIL
from torchvision.datasets.food101 import Food101
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple

class IMBALANCEFOOD(Food101):
    cls_num = 101

    def __init__(self, root, train=True, imb_type='exp', imb_factor=0.01, rand_number=0, transform=None, target_transform=None, download=False):
        super(IMBALANCEFOOD, self).__init__(root, split='train' if train else 'test', transform=transform, target_transform=target_transform, download=download)
        np.random.seed(rand_number)
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self._image_files) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'major':
            img_num_per_cls.append(int(img_max))
            for cls_idx in range(1, cls_num):
                num = img_max * imb_factor + (cls_num - 1.0 - cls_idx)
                img_num_per_cls.append(int(num))
        elif imb_type == 'poly':
            for cls_idx in range(cls_num):
                num = img_max * (1 / math.sqrt(cls_idx / (10 * imb_factor) + 1))
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_image_files = []
        new_labels = []
        targets_np = np.array(self._labels, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_image_files.extend([self._image_files[i] for i in selec_idx])
            new_labels.extend([the_class] * the_img_num)
        self._image_files = new_image_files
        self._labels = new_labels

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

