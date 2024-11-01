# general imports
import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import torchvision
import os
import pickle
import sys
import argparse
from torchvision import transforms, datasets
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import pandas as pd
#from my_utils import SavePlot, Plot_Calscores
import cv2
import numpy as np
import glob 
# from glob import glob

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter

from math import ceil
from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset,Dataset, Sampler

import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader

# My imports
sys.path.insert(0, './')
# import ICP.Score_Functions as scores

from ICP.utils_others import *
#from ICP.my_utils import *
#from ICP.clustering_utils import *
from ICP.conformal_utils2 import *

from collections import Counter
from scipy import stats, cluster

import models
from dataset.cifar10 import load_cifar10
from dataset.cifar100 import load_cifar100
# from dataset.Eurosat import load_Eurosat, EurosatDataset
# from dataset.EMNIST import load_Emnist
from train.imbalance_mini import IMBALANEMINIIMGNET
from train.imbalance_food import IMBALANCEFOOD
#from dataset.tiny_imagenet import load_Tiny

from ICP.generate_score import *

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (10, 7),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print(model_names, 'model_names')

    
# parameters
parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-s', '--splits', default=10, type=int, help='Number of experiments to estimate coverage')

parser.add_argument('--coverage_on_label', action='store_true', help='True for getting coverage and size for each label')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-ar', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--rho', default=0.01, type=float, help='imbalance factor')
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
parser.add_argument('--seeds', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('-score_functions', type=str,  nargs='+', 
                    help='Conformal score functions to use. List with a space in between. Options are'
                    '"softmax", "APS", "RAPS"')
parser.add_argument('-methods', type=str,  nargs='+', 
                    help='Conformal methods to use. List with a space in between. Options include'
                    '"MCP", "CCP", "k-CCP", "always_cluster"')
parser.add_argument('-seeds', type=int,  nargs='+', 
                    help='Seeds for random splits into calibration and validation sets,'
                    'List with spaces in between')
parser.add_argument('-avg_num_per_class', type=int,
                        help='Number of examples per class, on average, to include in calibration dataset')
parser.add_argument('--calibration_sampling', type=str, default='random',
                    help='How to sample the calibration set. Options are "random" and "balanced"')
parser.add_argument('--bins', type=int, default='10',
                    help='Histgram range to plot"')
parser.add_argument('--t_gap', type=float, default='0.9',
                    help='Concentration gap of truncated')
parser.add_argument('--c_gap', type=float, default='0.9',
                    help='Concentration gap of classwise')
parser.add_argument('--cl_gap', type=float, default='0.9',
                    help='Concentration gap of cluster')
parser.add_argument('--all', type=str, default='no',
                    help='Whether to compute all results')
parser.add_argument('--frac_clustering', type=float, default=-1,
                    help='For clustered conformal: the fraction of data used for clustering.'
                    'If frac_clustering and num_clusters are both -1, then a heuristic will be used to choose these values.')
parser.add_argument('--num_clusters', type=int, default=-1,
                    help='For clustered conformal: the number of clusters to request'
                    'If frac_clustering and num_clusters are both -1, then a heuristic will be used to choose these values.')

# parser.add_argument('--lmbda_val', type=float, default='0.01',
#                     help='lmbda val for RAPS')
# parser.add_argument('--k_reg', type=int, default='5',
#                     help='k_reg value for RAPS')

args = parser.parse_args()
print(f"args = {args}")
# parameters
alpha = args.alpha  # desired nominal marginal coverage
n_experiments = args.splits  # number of experiments to estimate coverage

dataset = args.dataset  # dataset to be used  CIFAR100', 'CIFAR10', 'ImageNet'
calibration_scores = ['SC', 'HCC', 'SC_Reg']  # score function to check 'HCC', 'SC', 'SC_Reg'
coverage_on_label = True # Whether to calculate coverage and size per class

# number of test points (if larger then available it takes the entire set)
if dataset == 'ImageNet':
    n_test = 50000
elif dataset == 'eurosat':
    n_test = 5400
elif dataset == 'EMNIST':
    n_test = 20800
elif dataset == 'tiny':
    n_test = 100000
else:
    n_test = 10000

# Validate parameters
assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'
assert isinstance(n_experiments, int) and n_experiments >= 1, 'number of splits must be a positive integer.'


# The GPU used for oue experiments can only handle the following quantities of images per batch
GPU_CAPACITY = args.batch_size
torch.cuda.set_device(args.gpu)
device = torch.cuda.current_device()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")
# set random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


base_path = "dataset={}/architecture={}/loss_type={}/imb_type={}/imb_factor={}/train_rule={}/epochs={}/batch-size={}\
        /lr={}/momentum={}/".format(args.dataset, args.arch, args.loss_type, args.imb_type, args.rho, args.train_rule,\
             args.epochs, args.batch_size, args.lr, args.momentum)
patha_model = '/checkpoint/' + base_path + "{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.arch, args.loss_type, \
    args.train_rule, args.imb_type, args.rho, args.rand_number) + '/ckpt.best.pth.tar'
patha_model = os.getcwd() + patha_model
#print(patha_model)

patha = './Results/new' + base_path 
patha_2 = './Results/' + base_path 
if not os.path.exists(patha):
    os.makedirs(patha)
# load datasets
if dataset == "cifar10":
    # Load train set
    num_classes = 10

    print(patha_model)
    checkpoint = torch.load(patha_model, map_location = 'cuda:0')
    best_acc = checkpoint['best_acc1']
    print(best_acc)
    num_classes = 10 if args.dataset == 'cifar10' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    #model = torch.load(patha_model).to(device)
    model.eval()

    train_transform = Compose([
        RandomCrop(32,padding=4),
        #Resize(image_size, BICUBIC),
        #RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124,117,104)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    test_transform = Compose([
        #Resize(image_size, BICUBIC),    
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True)

    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False,
        num_workers=args.workers, pin_memory=True)

elif dataset == 'tiny':
    # Load train set
    num_classes = 200


    checkpoint = torch.load(patha_model, map_location = 'cpu')
    best_acc = checkpoint['best_acc1']
    print(best_acc)
    num_classes = 200 if args.dataset == 'tiny' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    #model = torch.load(patha_model).to(device)
    model.eval()

    train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples = load_Tiny(
        patha + '/' + str(args.arch) + '_rho_' + str(args.rho) +'_ClassFrequency.png', 
        train_size=500, val_size=50,
        balance_val=True, batch_size=128,
        train_rho=args.rho, val_rho = args.rho,
        image_size=64, path='./data/tiny-imagenet-200')
 #   print(num_train_samples[0], num_train_samples[2], num_train_samples[3], num_train_samples[5], num_train_samples[6], num_train_samples[9])

elif dataset == "cifar100":

    num_classes = 100

  
    checkpoint = torch.load(patha_model, map_location = 'cpu')
    best_acc = checkpoint['best_acc1']
    print(best_acc)
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    #model = torch.load(patha_model).to(device)
    model.eval()
    
    train_transform = Compose([
        RandomCrop(32,padding=4),
        #Resize(image_size, BICUBIC),
        #RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124,117,104)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    test_transform = Compose([
        #Resize(image_size, BICUBIC),    
        ToTensor(),
        Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    train_dataset = CIFAR100(root='./data', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR100(root='./data', train=False, transform=test_transform, download=True)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False,
        num_workers=args.workers, pin_memory=True)


elif dataset == "ImageNet":
    # get dir of imagenet validation set
    imagenet_dir = "./datasets/imagenet"

    # ImageNet images pre-processing
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    # load dataset
    test_dataset = datasets.ImageFolder(imagenet_dir, transform)
elif dataset == 'EMNIST':

    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])
    num_classes = 27

    #patha_model = args.resume
    checkpoint = torch.load(patha_model, map_location = 'cuda:0')
    best_acc = checkpoint['best_acc1']
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm, features = 1)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

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

    train_dataset = EMNIST(root='./data', train=True, transform=train_transform, download=True, split = 'letters')
    val_dataset = EMNIST(root='./data', train=False, transform=test_transform, download=True, split = 'letters')
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False,
        num_workers=args.workers, pin_memory=True)

elif dataset == 'mini':

    num_classes = 100
    img_size = 64 if args.dataset == 'mini' else 32
    padding = 8 if args.dataset == 'mini' else 4

    #patha_model = args.resume
    checkpoint = torch.load(patha_model, map_location = 'cuda:0')
    best_acc = checkpoint['best_acc1']
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

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
            imb_factor=args.rho,
            rand_number=args.rand_number,
            transform=transform_train)
    val_dataset = IMBALANEMINIIMGNET(
            root_dir='data/mini-imagenet',
            csv_name="new_val.csv",
            json_path='data/mini-imagenet/classes_name.json',
            train=False,
            imb_type=args.imb_type,
            imb_factor=args.rho,
            rand_number=args.rand_number,
            transform=transform_val)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size= 300, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=val_dataset.collate_fn)

elif dataset == 'food':

    num_classes = 101

    #patha_model = args.resume
    checkpoint = torch.load(patha_model, map_location = 'cuda:0')
    best_acc = checkpoint['best_acc1']
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

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

    train_dataset = IMBALANCEFOOD(root='./data', train = True, imb_type=args.imb_type, imb_factor=args.rho, rand_number=args.rand_number, transform=transform_train, download=True)
    val_dataset = IMBALANCEFOOD(root='./data', train = False, imb_type=args.imb_type, imb_factor=args.rho, rand_number=args.rand_number, transform=transform_test, download=True)
    
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size= 250, shuffle=False,
        num_workers=args.workers, pin_memory=True)

else:
    print("No such dataset")
    exit(1)

# convert test set into tensor
# examples = enumerate(val_loader)
# batch_idx, (x_test, y_test) = next(examples)

# examples_cal = enumerate(eval_val_loader)
# batch_idx, (x_cal, y_cal) = next(examples_cal)

soft_max, ranks = get_softmax_and_ranks(val_loader, model, args)

test_dataset_with_ranks = DataWithRanks(val_dataset, soft_max, ranks)

data_item = test_dataset_with_ranks[0]  
ranks_test = data_item['rank']
#print(ranks)
softmax_scores = [item['softmax'] for item in test_dataset_with_ranks]
ranks = [item['rank'] for item in test_dataset_with_ranks]
labels = [item['label'] for item in test_dataset_with_ranks]
targets = [item['label'] for item in test_dataset_with_ranks]
#print(f'Creating dataset with softmax_scores and true label rank.')
softmax_scores_2 = np.array(softmax_scores)
#print(len(softmax_scores_2))
ranks_2 = np.array(ranks)
labels_2 = np.array(labels)
#
#if dataset == 'EMNIST':
#    acc_matrix = accuracy_matrix_EMNIST(val_loader, model, args, num_class = num_classes)
##print(acc_matrix[1])
#    print(f'Creating dataset with softmax_scores and true label rank.')
#else:
#    acc_matrix = accuracy_matrix(val_loader, model, args, num_class = num_classes)
#    #print(acc_matrix[1])
#    print(f'Creating dataset with softmax_scores and true label rank.')

#acc_matrix = np.array(acc_matrix)
# for i in range(10):
#     print(softmax_scores_2[i,:])
#     print(ranks_2[i])
#     print(labels_2[i])


dir_path = patha_2
filename = 'acc_matrix.npy'
full_path = os.path.join(dir_path, filename)

# Check and create the directory if it doesn't exist
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Check if the file exists in the specified directory
if not os.path.exists(full_path):
    acc_matrix = accuracy_matrix2(softmax_scores, targets, num_class = num_classes)
    acc_matrix = np.array(acc_matrix)
    
    # Save the computed matrix to the specified directory
    np.save(full_path, acc_matrix)

acc_matrix = np.load(full_path)
#print(acc_matrix)

def run_one_experiment(accuracy_matrix, softmax_scores, labels, ranks, dataset, save_folder, num_classes, alpha, bins, truncated_gap, class_gap, cluster_gap, n_totalcal, score_function_list, methods, seeds, 
                       cluster_args={'frac_clustering':'auto', 'num_clusters':'auto'},
                       save_preds=False, calibration_sampling='random', save_labels=False):
    '''
    Run experiment and save results
    
    Inputs:
        - dataset: string specifying dataset. Options are 'imagenet', 'cifar-100', 'places365', 'inaturalist'
        - n_totalcal: *average* number of examples per class. Calibration dataset is generated by sampling
          n_totalcal x num_classes examples uniformly at random
        - methods: List of conformal calibration methods. Options are 'standard', 'classwise', 
         'classwise_default_standard', 'cluster_proportional', 'cluster_doubledip','cluster_random'
         -cluster_args: Dict of arguments to be bassed into cluster_random
        - save_preds: if True, the val prediction sets are included in the saved outputs
        - calibration_sampling: Method for sampling calibration dataset. Options are 
        'random' or 'balanced'
        - save_labels: If True, save the labels for each random seed in {save_folder}seed={seed}_labels.npy
    '''
    #global method
    np.random.seed(0)
    
    #softmax_scores, labels = load_dataset(dataset)
    
    for score_function in score_function_list:
        curr_folder = os.path.join(save_folder, f'{calibration_sampling}_calset/n_totalcal={n_totalcal}/score={score_function}/')
        os.makedirs(curr_folder, exist_ok=True)

        print(f'====== score_function={score_function} ======')

        #print('Computing conformal score...')
        # if score_function == 'softmax':
        #     scores_all = 1 - softmax_scores
        # elif score_function == 'APS':
        #     scores_all = get_APS_scores_all(softmax_scores, randomize=True)
        # elif score_function == 'RAPS': 
        #     # RAPS hyperparameters (currently using ImageNet defaults)
        #     lmbda = .01 
        #     kreg = 5

        #     scores_all = get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True)
        # else:
        #     raise Exception('Undefined score function')
        # results_per_seed = {'Under Covered Ratio': [], 'Average Set Size': [], 'Method': []}
        
        for seed in seeds:
            print(f'\nseed={seed}')
            save_to = os.path.join(curr_folder, f'seed={seed}_allresults.pkl')

            # Always initialize an empty dictionary for all_results
            all_results = {}

            # If the results file exists, notify the user
            if os.path.exists(save_to):
                print(f'Results file for seed={seed} already exists. New results will overwrite the old ones.')

            print(save_to)
            unique_labels = np.unique(labels)
            #print(f"Unique labels in the dataset: {unique_labels}")

            for method in methods:
            # Split data
                if calibration_sampling == 'random':

                    if method == 'k-CCP' or method == 'k-CCP_T':
                        totalcal_scores, totalcal_labels, totalcal_ranks, val_scores, val_labels, true_val_ranks= truncated_random_split(softmax_scores, 
                                                                                                    labels, 
                                                                                                    ranks,
                                                                                                    n_totalcal, 
                                                                                                    seed=seed)

                    else:
                        totalcal_scores, totalcal_labels, val_scores, val_labels = random_split(softmax_scores, 
                                                                                                    labels, 
                                                                                                    n_totalcal, 
                                                                                                    seed=seed)
                    
                elif calibration_sampling == 'balanced':
                    #num_classes = scores_all.shape[1]

                    if method == 'k-CCP' or method == 'k-CCP_T':
                        totalcal_scores, totalcal_labels, totalcal_ranks, val_scores, val_labels, true_val_ranks= truncated_split_X_and_y(softmax_scores,
                                                                                                    labels, ranks,
                                                                                                    n_totalcal,num_classes,
                                                                                                    seed=seed, split='balanced')

                    else:
                        totalcal_scores, totalcal_labels, val_scores, val_labels = split_X_and_y_Orin(softmax_scores,
                                                                                                    labels, n_totalcal, num_classes, 
                                                                                                    seed=seed, split='balanced')
                
                #print(len(totalcal_scores_all))
                #print(len(totalcal_labels))
                #print(len(val_scores_all))
                #print(len(val_labels))

                else:
                    raise Exception('Invalid calibration_sampling option')
          
            # Inspect class imbalance of total calibration set
            cts = Counter(totalcal_labels).values()
            #print(f'Class counts range from {min(cts)} to {max(cts)}')

            val_rank = compute_val_score_rank(val_scores)

            if score_function == 'softmax':
                totalcal_scores_all = 1 - totalcal_scores
            elif score_function == 'APS':
                totalcal_scores_all = get_APS_scores_all(totalcal_scores, randomize=True)
                val_scores_all = get_APS_scores_all(val_scores, randomize=True)
            elif score_function == 'RAPS': 
                # RAPS hyperparameters (currently using ImageNet defaults)
                lmbda = lam 
                kreg = k_r
                totalcal_scores_all = get_RAPS_scores_all(totalcal_scores, lmbda, kreg, randomize=True)
                val_scores_all = get_RAPS_scores_all(val_scores, lmbda, kreg, randomize=True)
            else:
                raise Exception('Undefined score function')

            for method in methods:
                print(f'----- dataset={dataset}, n={n_totalcal},score_function={score_function}, seed={seed}, method={method} ----- ')

                if method == 'MCP':
                    # Standard conformal
                    all_results[method] = standard_conformal(totalcal_scores_all, totalcal_labels, 
                                                             val_scores_all, val_labels, alpha)

                elif method == 'CCP':
                    # Classwise conformal  
                    all_results[method] = classwise_conformal(totalcal_scores_all, totalcal_labels, 
                                                               val_scores_all, val_labels, alpha, class_gap, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat=np.inf, regularize=False)

                elif method == 'CCP_default_standard':
                    # Classwise conformal, but use standard qhat as default value instead of infinity 
                    all_results[method] = classwise_conformal(totalcal_scores_all, totalcal_labels, 
                                                               val_scores_all, val_labels, alpha, class_gap, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat='standard', regularize=False)
                elif method == 'k-CCP':
                    # Classwise conformal  
                    all_results[method] = truncated_classwise_conformal(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, 
                                                               val_scores_all, val_labels, val_rank, alpha, truncated_gap, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat=np.inf, regularize=False)
                elif method == 'k-CCP_T':
                    # Classwise conformal  
                    all_results[method] = truncated_classwise_conformal_test(accuracy_matrix, totalcal_scores_all, totalcal_labels, true_val_ranks, 
                                                               val_scores_all, val_labels, val_rank, alpha, class_gap, truncated_gap, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat=np.inf, regularize=False)
                elif method == 'k-CCP_default_standard':
                    # Classwise conformal, but use standard qhat as default value instead of infinity 
                    all_results[method] = truncated_classwise_conformal(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, 
                                                               val_scores_all, val_labels, alpha, truncated_gap, val_rank, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat='standard', regularize=False)
                elif method == 'regularized_k-CCP':
                    # Empirical-Bayes-inspired regularized classwise conformal (shrink class qhats to standard)
                    all_results[method] = truncated_classwise_conformal(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, 
                                                               val_scores_all, val_labels, alpha, truncated_gap, val_rank, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat='standard', regularize=True)
                
                elif method == 'exact_coverage_k-CCP':
                    # Apply randomization to qhats to achieve exact coverage
                    all_results[method] = truncated_classwise_conformal(accuracy_matrix, totalcal_scores_all, totalcal_labels, totalcal_ranks, 
                                                               val_scores_all, val_labels, alpha, truncated_gap, val_rank, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat=np.inf, regularize=False,
                                                               exact_coverage=True)

                elif method == 'cluster_proportional':
                    # Clustered conformal with proportionally sampled clustering set
                    all_results[method] = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                alpha, cluster_gap,
                                                                val_scores_all, val_labels, 
                                                                split='proportional')
                
                elif method == 'cluster_doubledip':
                    # Clustered conformal with double dipping for clustering and calibration
                    all_results[method] = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                               alpha,
                                                                val_scores_all, val_labels, 
                                                                split='doubledip')

                elif method == 'cluster_CP':
                    # Clustered conformal with double dipping for clustering and calibration
                    all_results[method] = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                alpha, cluster_gap, 
                                                                val_scores_all, val_labels, num_classes=val_scores_all.shape[1],
                                                                frac_clustering=cluster_args['frac_clustering'],
                                                                num_clusters=cluster_args['num_clusters'],
                                                                split='random')
                elif method == 'regularized_CCP':
                    # Empirical-Bayes-inspired regularized classwise conformal (shrink class qhats to standard)
                    all_results[method] = classwise_conformal(totalcal_scores_all, totalcal_labels, 
                                                               val_scores_all, val_labels, alpha, class_gap, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat='standard', regularize=True)
                
                elif method == 'exact_coverage_MCP':
                    # Apply randomization to qhat to achieve exact coverage
                    all_results[method] = standard_conformal(totalcal_scores_all, totalcal_labels,
                                                                            val_scores_all, val_labels, alpha,
                                                                            exact_coverage=True)
                    
                elif method == 'exact_coverage_CCP':
                    # Apply randomization to qhats to achieve exact coverage
                    all_results[method] = classwise_conformal(totalcal_scores_all, totalcal_labels, 
                                                               val_scores_all, val_labels, alpha, class_gap, 
                                                               num_classes=val_scores_all.shape[1],
                                                               default_qhat=np.inf, regularize=False,
                                                               exact_coverage=True)


                elif method == 'exact_coverage_cluster':
                    # Apply randomization to qhats to achieve exact coverage
                    all_results[method] = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                alpha,
                                                                val_scores_all, val_labels, 
                                                                frac_clustering=cluster_args['frac_clustering'],
                                                                num_clusters=cluster_args['num_clusters'],
                                                                split='random',
                                                                exact_coverage=True)

                else: 
                    raise Exception('Invalid method selected')
            
            # Optionally remove predictions from saved output to reduce memory usage
            if not save_preds:
                for m in all_results.keys():
                    all_results[m] = (all_results[m][0], all_results[m][1], all_results[m][2], all_results[m][3], all_results[m][4])
                    
            # Optionally save val labels
            if save_labels:
                save_labels_to = os.path.join(curr_folder, f'seed={seed}_labels.npy')
                np.save(save_labels_to, val_labels)
                print(f'Saved labels to {save_labels_to}')
            
            with open(save_to,'wb') as f:
                pickle.dump(all_results, f)
                print(f'Saved results to {save_to}')
            
            with open(save_to, 'rb') as f:
                all_results = pickle.load(f)

            data_lists = {'Prediction Set Size': [], 'Class Conditional Coverage': [], 'Condition on Class': [], 'Method': []}
            for method, result in all_results.items():
                data_lists['Prediction Set Size'].append(result[1]['Prediction Set Size'])
                data_lists['Class Conditional Coverage'].append(result[1]['Class Conditional Coverage'])
                data_lists['Condition on Class'].append(result[1]['Condition on Class'])
                data_lists['Method'].append(method)

            results = pd.DataFrame(data_lists)

            f_results = pd.DataFrame({
                'Prediction Set Size': np.concatenate(results['Prediction Set Size'].values),
                'Class Conditional Coverage': np.concatenate(results['Class Conditional Coverage'].values),
                'Condition on Class': np.concatenate(results['Condition on Class'].values),
                'Method': np.repeat(results['Method'].values, [len(x) for x in results['Prediction Set Size']])
            })
            
            # marginal_q = all_results['MCP'][0]
            # selected_methods = ["CCP"]
            # qhats_data = {'Class Quantile': [], 'Class Index': [], 'Method': []}
            # for method, result in all_results.items():
            #     if method in selected_methods:
            #         qhats_data['Class Quantile'].append(result[0]) 
            #         qhats_data['Class Index'].append(result[1]['Condition on Class'])
            #         qhats_data['Method'].append(method)

            # q_results = pd.DataFrame(qhats_data)

            # Q_results = pd.DataFrame({
            #     'Class Quantile': np.concatenate(q_results['Class Quantile'].values),
            #     'Class Index': np.concatenate(q_results['Class Index'].values),
            #     'Method': np.repeat(q_results['Method'].values, [len(x) for x in q_results['Class Quantile']])
            # })

            # for method, result in all_results.items():
            #     results_per_seed['Under Covered Ratio'].append(result[3]['Under Covered Ratio'])
            #     results_per_seed['Average Set Size'].append(result[2]['Average Set Size'])
            #     results_per_seed['Method'].append(method)

            # if num_classes <= 25:
                # SavePlot(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_type_{args.imb_type}_epoch_{args.epochs}_Covg.png'), num_classes, x = 'Condition on Class', y='Class Conditional Coverage', hue='Method', data=f_results, kind='point', legend=True, rotation=90)
                # SavePlot(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_type_{args.imb_type}_epoch_{args.epochs}_Size.png'), num_classes, x = 'Condition on Class', y='Prediction Set Size', hue='Method', data=f_results, kind='point', legend=False, rotation=90)
                # PlotHistgram(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_score_{args.score_functions}_epoch_{args.epochs}'), nbins = bins, results = f_results)
                # PlotLineGraph_quantile(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_type_{args.imb_type}_epoch_{args.epochs}'), results=Q_results)
                # SavePlotHistgram(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_score_{args.score_functions}_epoch_{args.epochs}'), x = 'Class Conditional Coverage', random = seed, nbins = bins, results = f_results)
                # SavePlotHistgram(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_score_{args.score_functions}_epoch_{args.epochs}'), x = 'Prediction Set Size', random = seed, nbins = bins, results = f_results)
                # SavePlot_Q_Histgram(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_score_{args.score_functions}_epoch_{args.epochs}'), x = 'Class Quantile', random = seed, nbins = bins, mq = marginal_q, results = Q_results)

            # else:
                # SavePlotHistgram(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_score_{args.score_functions}_epoch_{args.epochs}'), x = 'Class Conditional Coverage', random = seed, nbins = bins, results = f_results)
                # SavePlotHistgram(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_score_{args.score_functions}_epoch_{args.epochs}'), x = 'Prediction Set Size', random = seed, nbins = bins, results = f_results)
                # SavePlot_Q_Histgram(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_score_{args.score_functions}_epoch_{args.epochs}'), x = 'Class Quantile', random = seed, nbins = bins, mq = marginal_q, results = Q_results)
                # SavePlot(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_type_{args.imb_type}_epoch_{args.epochs}_Covg.png'), num_classes, x = 'Condition on Class', y='Class Conditional Coverage', hue='Method', data=f_results, kind='point', legend=True, rotation=90)
                # SavePlot(args, os.path.join(patha, f'{args.arch}_rho_{args.rho}_loss_{args.loss_type}_type_{args.imb_type}_epoch_{args.epochs}_Size.png'), num_classes, x = 'Condition on Class', y='Prediction Set Size', hue='Method', data=f_results, kind='point', legend=False, rotation=90)


def SavePlot(args, path, num_classes, x = None, y = None, col = None, hue = None, data = None, kind = None, legend = False, rotation = 45):
    #plt.style.use('seaborn')
    #print('ha ho66e')
    sns.set_style("whitegrid")

    font = 20
    sns.set(font_scale=1.5)
    s1 = sns.catplot(data = data, x = x, y = y, hue = hue, kind = kind, ci = 'sd', height = 4.5, aspect = 2.2, palette = ['black', 'lime', 'red'], legend = None, legend_out = False)
    axes = s1.axes.flatten()
    s1.set(xticklabels=[])

    if y == 'Class Conditional Coverage':
        for i, ax in enumerate(axes):
            ax.axhline(1-args.alpha, ls='--', c='blue')
            ax.legend(loc='upper left', title='method')
    
            # Retrieve legend handles and labels for the method (from hue)
            #handles, labels = ax.get_legend_handles_labels()
    
            #if handles:
                #ax.legend(handles=handles, labels=labels, loc='upper left', title='method', frameon=True, fontsize=13, fancybox=True, framealpha=0.9, bbox_to_anchor=(1, 1.2))
        min_ccc = data['Class Conditional Coverage'].min()
        # s1.set(ylim=(0.7, 1.05))
        s1.set(ylim=(min_ccc, 1.05))

    if y == 'Prediction Set Size':
        for i, ax in enumerate(axes):
            #s1.set_xticklabels(ax.get_xticklabels(), fontsize = 10, rotation = rotation)
            #s1.set_xticklabels(np.arange(0, num_classes+1, 5), fontsize = font)
            #print('ha')
            ax.legend(loc='upper left', title='method')
        #s1.set(ylim=(1, 100))


    plt.tight_layout()
    s1.savefig(path, dpi = 100, bbox_inches='tight', pad_inches=.1)
    plt.close('all')

def PlotHistgram(args, path, nbins, results=None):

    if not os.path.exists(path):
        os.makedirs(path)

    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)

    output_path = path

    methods = results['Method'].unique()

    # Only select the second and third methods
    selected_methods = methods[1:3]

    # Filter the data for the selected methods
    selected_data = results[results['Method'].isin(selected_methods)]

    # Get the minimum and maximum for 'Class Conditional Coverage' for the selected methods
    min_ccc = selected_data['Class Conditional Coverage'].min()
    max_ccc = selected_data['Class Conditional Coverage'].max()

    for method in selected_methods:
        plt.figure(figsize=(8, 6))
        sns.histplot(results[results['Method'] == method]['Class Conditional Coverage'], kde=True, bins = nbins, element='step', color='blue')
        plt.title(f'Frequency Histogram for {method}')
        plt.xlabel('Class Conditional Coverage')
        plt.ylabel('Frequency')

        # Add dotted line at 0.1cm width on x-axis
        plt.axvline(x=1-args.alpha, linestyle='--', color='red')

        # Adjust x-axis range
        plt.xlim(min_ccc, max_ccc)

        filename = f"{method}_histogram.png"
        plt.savefig(os.path.join(output_path, filename))
        plt.close()

    plt.figure(figsize=(10, 7))
    # Filter the data for combined histogram
    combined_data = results[results['Method'].isin(selected_methods)]
    sns.histplot(data=combined_data, x='Class Conditional Coverage', hue='Method', bins = nbins, element='step', common_norm=False, kde=True)

    # Add dotted line at 0.1cm width on x-axis
    plt.axvline(x=1-args.alpha, linestyle='--', color='red')

    # Adjust x-axis range
    plt.xlim(min_ccc, max_ccc)

    plt.title(f'Frequency Histogram for selected methods')
    plt.xlabel('Class Conditional Coverage')
    plt.ylabel('Frequency')
    filename = "combined_histogram_cov_selected_methods.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def SavePlotHistgram(args, path, x, random, nbins, results=None):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    sns.set_style("whitegrid")
    sns.set(font_scale= 1.5)

    methods = results['Method'].unique()

    # # Only select the second and third methods
    # selected_methods = methods[1:3]

    # # Filter the data for the selected methods
    # selected_data = results[results['Method'].isin(selected_methods)]
    if x == 'Class Conditional Coverage':

        # Get the minimum and maximum for 'Class Conditional Coverage' for the selected methods
        min_ccc = results['Class Conditional Coverage'].min()
        max_ccc = results['Class Conditional Coverage'].max()

        # plt.figure(figsize=(54, 30))
        # Filter the data for combined histogram
        ax = sns.histplot(data=results, x=x, hue='Method', bins = nbins, element='step', common_norm=False, kde=True, line_kws={"linewidth": 5}, legend=True)

        sns.move_legend(ax, "center left")

        # Add dotted line at 0.1cm width on x-axis
        plt.axvline(x=1-args.alpha, linestyle='--', color='red', label = '1 - alpha', linewidth=10)

        plt.setp(ax.get_legend().get_texts(), fontsize=32) # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize=32)

        # plt.title(f'Frequency Histogram of {args.dataset}', fontsize='72')
        plt.xlabel('Class Conditional Coverage', fontsize=48)
        # plt.ylabel('Frequency', fontsize=48)

        ax.set_xlabel('Class Conditional Coverage', fontsize=40)
        # ax.set_ylabel('Frequency', fontsize=48)

        ax.set_yticklabels(ax.get_yticks(), size = 32)
        plt.xticks([])

        filename = f"{args.dataset}_{args.imb_type}_{args.rho}_CovgHist_seed_{random}_score_{args.score_functions}_bins_{args.bins}.pdf"
        
    if x == 'Prediction Set Size':

        # Get the minimum and maximum for 'Class Conditional Coverage' for the selected methods
        min_ccc = results['Prediction Set Size'].min()
        max_ccc = results['Prediction Set Size'].max()

        # plt.figure(figsize=(75, 30))
        # Filter the data for combined histogram
        ax = sns.histplot(data=results, x=x, hue='Method', bins = nbins, element='step', common_norm=False, kde=True, line_kws={"linewidth": 5}, legend=True)

        # Add dotted line at 0.1cm width on x-axis
        # plt.axvline(x=1-args.alpha, linestyle='--', color='red')
        # plt.title(f'Frequency Histogram of {args.dataset}', fontsize= '72')
        plt.xlabel('Prediction Set Size', fontsize=48)
        # plt.ylabel('Frequency', fontsize=48)

        ax.set_xlabel('Prediction Set Size', fontsize=48)
        # ax.set_ylabel('Frequency', fontsize=48)

        plt.setp(ax.get_legend().get_texts(), fontsize=32) # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize=32)

        ax.set_yticklabels(ax.get_yticks(), size = 32)
        plt.xticks([])

        filename = f"{args.dataset}_{args.imb_type}_{args.rho}_SizeHist_seed_{random}_score_{args.score_functions}_bins_{args.bins}.pdf"


    # Adjust x-axis range
    plt.xlim(min_ccc, max_ccc)
    plt.savefig(os.path.join(directory, filename))
    plt.close()

def SavePlot_Q_Histgram(args, path, x, random, nbins, mq, results=None):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)

    min_ccc = results['Class Quantile'].min()
    max_ccc = results['Class Quantile'].max()

    # plt.figure(figsize=(48, 30))
    # Filter the data for combined histogram
    ax = sns.histplot(data=results, x=x, bins = nbins, element='step', common_norm=False, kde=True, line_kws={"linewidth": 5})

    # Add dotted line at 0.1cm width on x-axis
    plt.axvline(x=mq, linestyle='--', color='red', label = 'marginal quantile', linewidth=5)

    # Display the legend
    plt.legend(fontsize=32)

    # plt.title(f'Frequency Histogram of {args.dataset}', fontsize='72')
    plt.xlabel('Class Quantile', fontsize=48)
    # plt.ylabel('Frequency', fontsize=48)

    ax.set_yticklabels(ax.get_yticks(), size = 32)
    plt.xticks([])

    ax.set_xlabel('Class Quantile', fontsize=48)
    # ax.set_ylabel('Frequency', fontsize=48)

    filename = f"{args.dataset}_{args.imb_type}_{args.rho}_QuanHist_seed_{random}_cgap_{args.c_gap}_bins_{args.bins}.pdf"

    # Adjust x-axis range
    plt.xlim(min_ccc, max_ccc)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


def PlotLineGraph_quantile(args, path, results=None):
    # Make sure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set seaborn style and scale
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)

    # Use seaborn's lineplot function to create the plot
    sns.lineplot(data=results, x='Class Index', y='Class Quantile', hue='Method')  

    # Set the plot title and labels
    plt.title(f'Line Graph for Class Quantile')
    plt.xlabel('Class index')
    plt.ylabel('Class Quantile')  

    plt.tight_layout()

    # Construct filename and save the plot
    filename = "linegraph_quantile.png"
    plt.savefig(os.path.join(directory, filename))
    plt.close()

# def PlotHistgram(args, path, results = None):

#     if not os.path.exists(path):
#         os.makedirs(path)

#     sns.set_style("whitegrid")
#     sns.set(font_scale=1.5)

#     output_path = path

#     methods = results['Method'].unique()

#     # Get global minimum and maximum for 'Class Conditional Coverage'
#     min_ccc = results['Class Conditional Coverage'].min()
#     max_ccc = results['Class Conditional Coverage'].max()

#     for method in methods:
#         plt.figure(figsize=(8, 6))
#         sns.histplot(results[results['Method'] == method]['Class Conditional Coverage'], kde=True, color='blue')
#         plt.title(f'Frequency Histogram for {method}')
#         plt.xlabel('Class Conditional Coverage')
#         plt.ylabel('Frequency')
    
#         # Add dotted line at 0.1cm width on x-axis
#         plt.axvline(x=0.1, linestyle='--', color='red')
    
#         # Adjust x-axis range
#         plt.xlim(min_ccc, max_ccc)
    
#         filename = f"{method}_histogram.png"
#         plt.savefig(os.path.join(output_path, filename))
#         plt.close()

#     plt.figure(figsize=(10, 7))
#     sns.histplot(data=results, x='Class Conditional Coverage', hue='Method', element='step', common_norm=False, kde=True)

#     # Add dotted line at 0.1cm width on x-axis
#     plt.axvline(x=1-args.alpha, linestyle='--', color='red')

#     # Adjust x-axis range
#     plt.xlim(min_ccc, max_ccc)

#     plt.title(f'Frequency Histogram for all methods')
#     plt.xlabel('Class Conditional Coverage')
#     plt.ylabel('Frequency')
#     plt.legend(title='Method')
#     filename = "combined_histogram.png"
#     plt.savefig(os.path.join(output_path, filename))
#     plt.close()


# Helper function                
def initialize_metrics_dict(methods):
    
    metrics = {}
    for method in methods:
        metrics[method] = {'Under Coverage Ratio': [],
                           # 'Average Set Size': [],
                           # 'avg_set_size': [],
                           # 'marginal_cov': [],
                           'Avg Set Size': []} # Could also retrieve other metrics
        
    return metrics


def average_results_across_seeds(folder, print_results=True, show_seed_ct=False, 
                                 methods=['MCP', 'CCP', 'k-CCP'],
                                 max_seeds=np.inf):
    '''
    Input:
        - max_seeds: If we discover more than max_seeds random seeds, only use max_seeds of them
    '''

    
    file_names = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    num_seeds = len(file_names)
    # print(f"num_seeds:{num_seeds}")
    if show_seed_ct:
        print('Number of seeds found:', num_seeds)
    if max_seeds < np.inf and num_seeds > max_seeds:
        print(f'Only using {max_seeds} seeds')
        file_names = file_names[:max_seeds]
    
    metrics = initialize_metrics_dict(methods)
    
    for pth in file_names:
        with open(pth, 'rb') as f:
            results = pickle.load(f)
                        
        for method in methods:
            try:
                metrics[method]['Under Coverage Ratio'].append(results[method][2]['undercov ratio'])
                # metrics[method]['avg_set_size'].append(results[method][3]['mean'])
                # metrics[method]['max_class_cov_gap'].append(results[method][2]['max_gap'])
                # metrics[method]['marginal_cov'].append(results[method][2]['marginal_cov'])
                metrics[method]['Avg Set Size'].append(results[method][2]['average set size'])
            except Exception as e:
                print(f'Missing {method} in {pth}. Error: {e}')


    # print(metrics)

    under_undercovered_means = []
    under_undercovered_std = []
    set_size_means = []
    set_size_std = []
    
    for method in methods:
        n = num_seeds
        under_undercovered_means.append(np.mean(metrics[method]['Under Coverage Ratio']))
        under_undercovered_std.append(np.std(metrics[method]['Under Coverage Ratio'])/np.sqrt(n))

        set_size_means.append(np.mean(metrics[method]['Avg Set Size']))
        set_size_std.append(np.std(metrics[method]['Avg Set Size'])/np.sqrt(n))

        if print_results:
            print(f"  {method}:"
                  f" Under_ratio_mean: {np.mean(metrics[method]['Under Coverage Ratio'])}", 
                  f"Under_ratio_std: {np.std(metrics[method]['Under Coverage Ratio'])/np.sqrt(n)}", 
                  f"Avg_size_mean: {np.mean(metrics[method]['Avg Set Size'])}", 
                  f"Avg_size_std: {np.std(metrics[method]['Avg Set Size'])/np.sqrt(n)}")
        
        
    df = pd.DataFrame({'method': methods,
                      'avg_set_size_mean': set_size_means,
                      'avg_set_size_std': set_size_std,
                      'undercovered_mean': under_undercovered_means,
                      'undercovered_std': under_undercovered_std})
    
    # if display_table:
    #     display(df) # For Jupyter notebooks
        
    return df

# Helper function for get_metric_df
def initialize_dict(metrics, methods, suffixes=['mean', 'std']):
    d = {}
    for suffix in suffixes: 
        for metric in metrics:
            d[f'{metric}_{suffix}'] = {}

            for method in methods:

                d[f'{metric}_{suffix}'][method] = []
            
            
    return d

def get_metric_df(dataset, cal_sampling, metric, 
                  score_function,
                  method_list = ['standard', 'classwise', 'truncated'],
                  n_list = [500, 50, 250],
                  show_seed_ct=False,
                  print_folder=True,
                  save_folder=patha): # May have to update this path
    '''
    Similar to average_results_across_seeds
    '''
    
    aggregated_results = initialize_dict([metric], method_list)

    for n_totalcal in n_list:

        curr_folder = f'{save_folder}/{dataset}/{cal_sampling}_calset/n_totalcal={n_totalcal}/score={score_function}/score={score_function}'
        if print_folder:
            print(curr_folder)

        df = average_results_across_seeds(curr_folder, print_results=False, 
                                          display_table=False, methods=method_list, max_seeds=10,
                                          show_seed_ct=show_seed_ct)

        for method in method_list:

            for suffix in ['mean', 'se']: # Extract mean and SE

                aggregated_results[f'{metric}_{suffix}'][method].append(df[f'{metric}_{suffix}'][df['method']==method].values[0])
  
    return aggregated_results

# Not used in paper
def plot_class_coverage_histogram(folder, path, desired_cov=None, vmin=.6, vmax=1, nbins=30, 
                                  title=None, methods=['standard', 'classwise', 'truncated']):
    '''
    For each method, aggregate class coverages across all random seeds and then 
    plot density/histogram. This is equivalent to estimating a density for each
    random seed individually then averaging. 
    
    Inputs:
    - folder: (str) containing path to folder of saved results
    - desired_cov: (float) Desired coverage level 
    - vmin, vmax: (floats) Specify bin edges 
    - 
    '''
    sns.set_style(style='white', rc={'axes.spines.right': False, 'axes.spines.top': False})
    sns.set_palette('pastel')
    sns.set_context('talk') # 'paper', 'talk', 'poster'
    
    # For plotting
    map_to_label = {'standard': 'Standard', 
                   'classwise': 'Classwise',
                   'truncated': 'Truncated',}
    map_to_color = {'standard': 'gray', 
                   'classwise': 'lightcoral',
                   'truncated': 'dodgerblue'}
    
    bin_edges = np.linspace(vmin,vmax,nbins+1)
    
    file_names = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    num_seeds = len(file_names)
    print('Number of seeds found:', num_seeds)
    
    # OPTION 1: Plot average with 95% CIs
    cts_dict = {}
    for method in methods:
        cts_dict[method] = np.zeros((num_seeds, nbins))
        
    for i, pth in enumerate(file_names):
        with open(pth, 'rb') as f:
            results = pickle.load(f)
            
        for method in methods:
            
            cts, _ = np.histogram(results[method][2]['class-conditional coverage'], bins=bin_edges)
            cts_dict[method][i,:] = cts
    
    for method in methods:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        graph = sns.lineplot(x=np.tile(bin_centers, num_seeds), y=np.ndarray.flatten(cts_dict[method]),
                     label=map_to_label[method], color=map_to_color[method])

    if desired_cov is not None:
        graph.axvline(desired_cov, color='black', linestyle='dashed', label='Desired coverage')
        
    plt.xlabel('Class-conditional coverage')
    plt.ylabel('Number of classes')
    plt.title(title)
    plt.ylim(bottom=0)
    plt.xlim(right=vmax)
    plt.legend()
    #plt.show()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    
    # OPTION 2: Plot average, no CIs
#     class_coverages = {}
#     for method in methods:
#         class_coverages[method] = []
        
#     for pth in file_names:
#         with open(pth, 'rb') as f:
#             results = pickle.load(f)
            
#         for method in methods:
#             class_coverages[method].append(results[method][2]['raw_class_coverages'])
    
#     bin_edges = np.linspace(vmin,vmax,30) # Can adjust
    
#     for method in methods:
#         aggregated_scores = np.concatenate(class_coverages[method], axis=0)
#         cts, _ = np.histogram(aggregated_scores, bins=bin_edges, density=False)
#         cts = cts / num_seeds 
#         plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, cts, '-o', label=method, alpha=0.7)
        
#     plt.xlabel('Class-conditional coverage')
#     plt.ylabel('Number of classes')
#     plt.legend()

#     # OPTION 3: Plot separate lines
#     class_coverages = {}
#     for method in methods:
#         class_coverages[method] = []
        
#     for pth in file_names:
#         with open(pth, 'rb') as f:
#             results = pickle.load(f)
            
#         for method in methods:
#             class_coverages[method].append(results[method][2]['raw_class_coverages'])
    
#     bin_edges = np.linspace(vmin,vmax,30) # Can adjust
    
#     for method in methods:
#         for class_covs in class_coverages[method]:
#             cts, _ = np.histogram(class_covs, bins=bin_edges, density=False)
#             plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, cts, '-', alpha=0.3,
#                      label=map_to_label[method], color=map_to_color[method])
        
#     plt.xlabel('Class-conditional coverage')
#     plt.ylabel('Number of classes')
#     plt.show()
#     plt.legend()

# For square-root scaling in plots
import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import numpy as np

class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """
 
    name = 'squareroot'
 
    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)
 
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())
 
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax
 
    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform_non_affine(self, a): 
            return np.array(a)**0.5
 
        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()
 
    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform(self, a):
            return np.array(a)**2
 
        def inverted(self):
            return SquareRootScale.SquareRootTransform()
 
    def get_transform(self):
        return self.SquareRootTransform()
 
#mscale.register_scale(SquareRootScale)

run_one_experiment(accuracy_matrix = acc_matrix, softmax_scores = softmax_scores_2, labels = labels_2, ranks = ranks_2, dataset = args.dataset, save_folder = patha, num_classes = num_classes, alpha = args.alpha, bins = args.bins, truncated_gap = args.t_gap, class_gap = args.c_gap, cluster_gap = args.cl_gap, n_totalcal = args.avg_num_per_class, score_function_list = args.score_functions, methods = args.methods, seeds = args.seeds, 
                       cluster_args={'frac_clustering':'auto', 'num_clusters':'auto'},
                       save_preds=False, calibration_sampling=args.calibration_sampling, save_labels=False)
#plot_class_coverage_histogram(paths, patha + '/' + str(args.arch) + '_rho_' + str(args.rho) + '_class_coverage_histogram.png', desired_cov=args.alpha, vmin=.6, vmax=1, nbins=30, 
                                  #title='Class Coverage Histogram', methods=['standard', 'classwise', 'truncated'])
if args.all == 'yes':
    result_folder = os.path.join(patha, f'{args.calibration_sampling}_calset/n_totalcal={args.avg_num_per_class}/score={args.score_functions}/')
    # Remove square brackets from the score part and add trailing slash
    result_folder = result_folder.replace("score=['", "score=").replace("']", "/")
    # print(result_folder)
    table_results = average_results_across_seeds(folder = result_folder, print_results=True, show_seed_ct=True, methods=['MCP', 'CCP', 'k-CCP', 'cluster_CP'], max_seeds=np.inf)
    output_path = os.path.join(patha, f'{args.dataset}_{args.arch}_rho_{args.rho}_loss_{args.loss_type}_type_{args.imb_type}_score_{args.score_functions}_total_results.csv')
    table_results.to_csv(output_path, index=False)
else:
    None
