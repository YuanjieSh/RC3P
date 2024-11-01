import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import math

class IMBALANEMINIIMGNET(Dataset):
    cls_num = 100

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 train = True, 
                 imb_type='exp', 
                 imb_factor=0.01, 
                 rand_number=0,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.targets = self.img_label.copy()
        self.samples = list(zip(self.img_paths, self.img_label)) 
        self.labels = set(csv_data["label"].values)
        self.transform = transform

        np.random.seed(rand_number)
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
        else:
            None

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
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
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            #print(self.samples)
            res_list = [self.samples[i] for i in selec_idx]
            #print(res_list)
            new_data.extend(res_list)
            new_targets.extend([the_class, ] * the_img_num)
        #new_data = np.vstack(new_data)
        self.samples = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

