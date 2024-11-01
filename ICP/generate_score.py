import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch.nn.functional import softmax
from collections import defaultdict
from sklearn.metrics import top_k_accuracy_score
import torch.nn.functional as F

import os
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_softmax_and_ranks(val_loader, model, args):

    model.eval()
    logits_all = []  # Initialize list for logits
    logit_ranks_all = []  # Initialize list for logit ranks
    print(f'Computing the true label rank in softmax scores).')

    with torch.no_grad():  # Turn off gradients, as we are in test mode
        for x, targets in tqdm(val_loader):  

            x = x.to(args.gpu, non_blocking=True)  # Move inputs to GPU
            targets = targets.to(args.gpu, non_blocking=True)  # Move labels to GPU

            # Forward pass
            outputs = model(x)  # This gets the logits
            softmax_scores = F.softmax(outputs, dim=1).cpu().numpy()

            # Get the ranks
            _, indices = torch.sort(outputs, descending=True)
            ranks = torch.zeros_like(indices)
            for i in range(outputs.shape[0]):
                ranks[i][indices[i]] = torch.arange(outputs.shape[1], device='cuda')

            logit_ranks = ranks[torch.arange(outputs.shape[0]), targets].detach().cpu().numpy()
            logits = softmax_scores

            logits_all.append(logits)
            logit_ranks_all.append(logit_ranks)

    logits_all = np.concatenate(logits_all, axis=0)
    logit_ranks_all = np.concatenate(logit_ranks_all, axis=0)

    return logits_all, logit_ranks_all

class DataWithRanks(torch.utils.data.Dataset):
    def __init__(self, dataset, softmax_scores, ranks):
        self.dataset = dataset
        self.softmax_scores = softmax_scores
        self.ranks = ranks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        softmax_scores = self.softmax_scores[idx]
        rank = self.ranks[idx]
        #print(f'Creating dataset with softmax_scores and true label rank.')
        return {'image': image, 'label': label, 'softmax': softmax_scores, 'rank': rank}


def calc_top_k_accuracy_per_class(val_loader, model, args, k):
    model.eval()

    class_targets = defaultdict(list)
    class_scores = defaultdict(list)

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            softmax_scores = F.softmax(output, dim=1).cpu().numpy()

            # Divide the targets and scores into classes
            for score, t in zip(softmax_scores, target.cpu().numpy()):
                class_targets[t].append(t)
                class_scores[t].append(score)

    # Compute top-k accuracy for each class
    top_k_acc_per_class = {}
    for class_label in class_targets.keys():
        targets = class_targets[class_label]
        scores = class_scores[class_label]

        # Compute top-k accuracy manually
        correct = 0
        for target, score in zip(targets, scores):
            if target in np.argsort(score)[-k:]:
                correct += 1

        top_k_acc = correct / len(targets)
        top_k_acc_per_class[class_label] = top_k_acc

    return top_k_acc_per_class

def accuracy_matrix(val_loader, model, args, num_class):
    matrix = []
    for k in range(1, num_class+1):
        cls_test_2 = calc_top_k_accuracy_per_class(val_loader, model, args, k)
    
        cls_test_3 = [cls_test_2[i] for i in range(len(cls_test_2))]

        matrix.append(cls_test_3)

    return matrix

def accuracy_matrix_EMNIST(val_loader, model, args, num_class):
    matrix = []
    for k in range(1, num_class):
        cls_test_2 = calc_top_k_accuracy_per_class(val_loader, model, args, k)
    
        cls_test_3 = [cls_test_2[i] for i in cls_test_2.keys()]

        matrix.append(cls_test_3)

    return matrix

def calc_top_k_accuracy_per_class2(softmax_scores, targets, num_class, k):
    class_targets = defaultdict(list)
    class_scores = defaultdict(list)

    # Divide the targets and scores into classes
    for score, t in zip(softmax_scores, targets):
        class_targets[t].append(t)
        class_scores[t].append(score)

    # Compute top-k accuracy for each class
    top_k_acc_per_class = {}
    for class_label in class_targets.keys():
        targets = class_targets[class_label]
        scores = class_scores[class_label]

        # Compute top-k accuracy manually
        correct = 0
        for target, score in zip(targets, scores):
            if target in np.argsort(score)[-k:]:
                correct += 1

        top_k_acc = correct / len(targets)
        top_k_acc_per_class[class_label] = top_k_acc

    return top_k_acc_per_class

def accuracy_matrix2(softmax_scores, targets, num_class):
    matrix = []
    for k in range(1, num_class+1):
        cls_test_2 = calc_top_k_accuracy_per_class2(softmax_scores, targets, num_class, k)
    
        cls_test_3 = [cls_test_2[i] for i in range(len(cls_test_2))]

        matrix.append(cls_test_3)

    return matrix