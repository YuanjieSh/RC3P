from math import ceil
from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset,Dataset, Sampler

import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader
import numpy as np
import random
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Acrually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

def load_Emnist(save_path = None, test_size1 = None, test_size2 = None, train_size=4000,train_rho=0.01,val_size=1000,val_rho=0.01,image_size=32,batch_size=128,num_workers=4,path='./data',num_classes=27,balance_val=False):
    train_transform = Compose([
        RandomCrop(28,padding=4),
        #Resize(image_size, BICUBIC),
        #RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124,117,104)),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    test_transform = Compose([
        #Resize(image_size, BICUBIC),    
        ToTensor()
    ])

    train_dataset = EMNIST(root=path, train=True, transform=train_transform, download=True, split = 'letters')
    test_dataset = EMNIST(root=path, train=False, transform=test_transform, download=True, split = 'letters')
    train_x,train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
    test_x, test_y = np.array(test_dataset.data), np.array(test_dataset.targets)
    total_size=4600
    #print(train_y.min(), train_y.max(), test_y.min(), test_y.max(), 's1')
    #train_y = train_y - 1
    #test_y = test_y - 1
    #print(train_y.min(), train_y.max(), test_y.min(), test_y.max(), 's1')
    #exit(1)
    num_total_samples=[]
    num_train_samples=[]
    num_val_samples=[]
    num_test_sample1 = []
    num_test_sample2 = []

    test_mu = train_rho**(1./(num_classes-1))
    test_index1 = []
    test_index2 = []
    for i in range(num_classes):
        num_test_sample1.append(ceil(test_size1*(test_mu**i)))
        num_test_sample2.append(ceil(test_size1*(test_mu**i)))

    for i in range(num_classes):
        test_index1.extend(np.where(test_y == i)[0][:num_test_sample1[i]])
        test_index2.extend(np.where(test_y == i)[0][-num_test_sample2[i]:])

    hyper_data,hyper_targets=test_x[test_index1],test_y[test_index1]
    test_data,test_targets=test_x[test_index2],test_y[test_index2]
    #print(test_index1)
    #print('ss')
    #print(test_index2)
    #exit(1)

    if not balance_val:
        train_mu=train_rho**(1./(num_classes-1))
        val_mu=val_rho**(1./(num_classes-1))
        for i in range(num_classes):
            num_total_samples.append(ceil(total_size*(train_mu**i)))
            num_train_samples.append(ceil(train_size*(train_mu**i)))
            num_val_samples.append(ceil(val_size*(val_mu**i)))
            #num_val_samples.append(num_total_samples[-1]-num_train_samples[-1])
            #num_val_samples.append(round(val_size*(val_mu**i)))
    elif balance_val:
        train_mu=train_rho**(1./(num_classes-1))
        for i in range(num_classes):
            num_val_samples.append(val_size)
            num_total_samples.append(ceil(total_size*(train_mu**i)))
            num_train_samples.append(ceil(train_size*(train_mu**i)))
            #num_train_samples.append(num_total_samples[-1]-num_val_samples[-1])

    train_index=[]
    val_index=[]
    #print(train_x,train_y)
    #print(num_train_samples,num_val_samples)

    #for i in range(num_classes):
    #    print(len(np.where(test_y==i)[0]))

    #exit(1)



    for i in range(num_classes):
        train_index.extend(np.where(train_y==i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y==i)[0][-num_val_samples[i]:])
    #print(val_index)
    #exit(1)
    total_index=[]
    total_index.extend(train_index)
    total_index.extend(val_index)
    total_index=list(set(total_index))
    random.shuffle(total_index)
    train_x, train_y=train_x[total_index], train_y[total_index]

    train_index=[]
    val_index=[]
    #print(train_x,train_y)
    print(f"train histogram: {num_train_samples}, val histogram: {num_val_samples}")
    print(f"length; train histogram: {len(num_train_samples)}, val histogram: {len(num_val_samples)}")


    #print(df)
    for i in range(num_classes):
        train_index.extend(np.where(train_y==i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y==i)[0][-num_val_samples[i]:])

    random.shuffle(train_index)
    random.shuffle(val_index)
    
    train_data,train_targets=train_x[train_index],train_y[train_index]
    val_data,val_targets=train_x[val_index],train_y[val_index]
    #print(f" train: {len(train_data)}, val: {len(val_data)} hyper: {len(hyper_data)}, test: {len(test_data)}")

    #exit(1)
    #print(val_targets.min(), val_targets.max(), train_targets.min(), train_targets.max(), 's1')
    #exit(1)
    train_dataset = CustomDataset(train_data,train_targets,train_transform)
    val_dataset = CustomDataset(val_data,val_targets,train_transform)
    train_eval_dataset = CustomDataset(train_data,train_targets,test_transform)
    val_eval_dataset = CustomDataset(val_data,val_targets,test_transform)


    HYPER_DATA = CustomDataset(hyper_data,hyper_targets,test_transform)
    TEST_DATA = CustomDataset(test_data,test_targets,test_transform)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=True, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_data), num_workers=num_workers, 
                            shuffle=True, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=20800, num_workers=num_workers, 
                            shuffle=False, drop_last=False, pin_memory=True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=len(train_data), num_workers=num_workers, 
                                shuffle=False, drop_last=False, pin_memory=True)
    eval_val_loader = DataLoader(val_eval_dataset, batch_size=len(val_data), num_workers=num_workers, 
                                shuffle=False, drop_last=False, pin_memory=True)

    HYPER_loader = DataLoader(HYPER_DATA, batch_size=len(hyper_data), num_workers=num_workers, 
                        shuffle=False, drop_last=False, pin_memory=True)
    TEST_loader = DataLoader(TEST_DATA, batch_size=len(test_data), num_workers=num_workers, 
                            shuffle=False, drop_last=False, pin_memory=True)

    return train_loader,val_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples, HYPER_loader, TEST_loader, test_loader

class CustomDataset(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)
#load_cifar10()