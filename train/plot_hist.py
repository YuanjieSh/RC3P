import argparse
import os
import pandas as pd
import random
import time
import warnings
import sys
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from imbalance_cifar10 import IMBALANCECIFAR10, IMBALANCECIFAR100
from imbalance_mini import IMBALANEMINIIMGNET
from imbalance_food import IMBALANCEFOOD
from utils import *
from losses import LDAMLoss, FocalLoss

from torch.nn.functional import softmax
from collections import defaultdict
from sklearn.metrics import top_k_accuracy_score
import torch.nn.functional as F


sys.path.insert(0, './')

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar100', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
best_acc1 = 0


def main():
    args = parser.parse_args()
    base_path = "dataset={}/architecture={}/loss_type={}/imb_type={}/imb_factor={}/train_rule={}/epochs={}/batch-size={}\
        /lr={}/momentum={}/".format(args.dataset, args.arch, args.loss_type, args.imb_type, args.imb_factor, args.train_rule,\
             args.epochs, args.batch_size, args.lr, args.momentum)
    args.root_log =  'log/' + base_path
    args.root_model = 'checkpoint/' + base_path
    #print(args.root_model)
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    #print(args.store_name)
    result_path = './Results/' + base_path

    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args, result_path)

def main_worker(gpu, ngpus_per_node, args, result_path):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.dataset == 'cifar10':
        num_classes = 10 

        cudnn.benchmark = True

        # Data loading code

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        
    elif args.dataset == 'cifar100':

        num_classes = 100 

        cudnn.benchmark = True

        # Data loading code

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
        
    elif args.dataset == 'mini':
        num_classes = 100 
        img_size = 64 if args.dataset == 'mini' else 32
        padding = 8 if args.dataset == 'mini' else 4
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomCrop(img_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = IMBALANEMINIIMGNET(
            root_dir='data/mini-imagenet',
            csv_name="new_train.csv",
            json_path='data/mini-imagenet/classes_name.json',
            train=True,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            rand_number=args.rand_number,
            transform=transform_train)
        val_dataset = IMBALANEMINIIMGNET(
            root_dir='data/mini-imagenet',
            csv_name="new_val.csv",
            json_path='data/mini-imagenet/classes_name.json',
            train=False,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            rand_number=args.rand_number,
            transform=transform_val)

        cls_num_list = train_dataset.get_cls_num_list()

    elif args.dataset == 'food':
        num_classes = 101

        img_size = 224 if args.dataset == 'food' else 32
        padding = 8 if args.dataset == 'food' else 4

        input_size = 224

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(35),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomPosterize(bits=2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = IMBALANCEFOOD(root='./data', train = True, imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, transform=transform_train, download=True)
        val_dataset = IMBALANCEFOOD(root='./data', train = False, imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, transform=transform_test, download=True)
        
    else:
        warnings.warn('Dataset is not listed')
        return


    cls_num_list = train_dataset.get_cls_num_list()
    class_indices = range(num_classes) 
    print('cls num list:')
    print(cls_num_list)
    # create a DataFrame
    results = pd.DataFrame(list(zip(class_indices, cls_num_list)), columns=['class', 'count'])

    # pass the DataFrame to the SavePlot function
    SavePlot(args, result_path, results)


def SavePlot(args, path, results=None):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)

    plt.figure(figsize=(10, 7))

    sns.barplot(x='class', y='count', data=results)

    plt.xticks([])

    plt.title(f'Data Distribution of {args.dataset}')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Data')

    filename = f"Bar_data_{args.dataset}_type_{args.imb_type}.pdf"

    plt.savefig(os.path.join(directory, filename))
    plt.close()

if __name__ == '__main__':
    main()



