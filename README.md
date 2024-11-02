## Conformal training

This repository contains a implementation of **RC3P**
corresponding to the follow paper:

Yuanjie Shi, Subhankar Ghosh, Taha Belkhouja, Janardhan Rao Doppa, Yan Yan.
*[Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration](
https://openreview.net/pdf?id=T7dS1Ghwwu)*.
NeurIPS, 2024.

## Overview

Conformal prediction (CP) is an emerging uncertainty quantification framework that allows us to construct a prediction set to cover the true label with a pre-specified marginal or conditional probability.
Although the valid coverage guarantee has been extensively studied for classification problems, CP often produces large prediction sets which may not be practically useful.
This issue is exacerbated for the setting of class-conditional coverage on classification tasks with many and/or imbalanced classes.
This paper proposes the Rank Calibrated Class-conditional CP (RC3P) algorithm to reduce the prediction set sizes to achieve class-conditional coverage, where the valid coverage holds for each class.
In contrast to the standard class-conditional CP (CCP) method that uniformly thresholds the class-wise conformity score for each class, the augmented label rank calibration step allows RC3P to selectively iterate this class-wise thresholding subroutine only for a subset of classes whose class-wise top-k error is small.
We prove that agnostic to the classifier and data distribution, RC3P achieves class-wise coverage. We also show that RC3P reduces the size of prediction sets compared to the CCP method. 
Comprehensive experiments on multiple real-world datasets demonstrate that RC3P achieves class-wise coverage and 26.25% reduction in prediction set sizes on average.

## Contents
The major content of our repo are:
 - `checkpoint/` Our pre trained models.
 - `data/` A folder that contains the datasets used in our experiments CIFAR10, CIFAR100, eorosat, EMNIST.
 - `dataset/` Manually form dataloader.
 - `ICP/` The main folder containing the python scripts for running the experiments for ICP.
 - `log/` Details pre trained models.
 - `models/` Contains ResNet networks.
 - `Results/` A folder that contains different files from different experiments.
 - `test/` Contains all training codes.
 - `train/` Contains all training codes.

ICP folder contains:

1. `main_new3.py`: the main code for running experiments for main results.
2. `main_test3.py`: the main code for running experiments for abalation study.

## Prerequisites

Prerequisites for running our code:
 - numpy
 - scipy
 - sklearn
 - torch
 - tqdm
 - seaborn
 - torchvision
 - pandas
 - plotnine
 
## Running instructions
1.  Install dependencies:
```
conda create -n ICP python=3.8
conda activate ICP
conda install -c conda-forge numpy
conda install -c conda-forge scipy
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge seaborn
conda install -c conda-forge pandas
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge plotnine
```

### Training 


- To produce trained models for different settings of the CIFAR10 data; please run the following command.
```
chmod +x train/CIFAR10_train.sh
./train/CIFAR10_train.sh
```

- To produce trained models for different settings of the CIFAR100 data; please run the following command.
```
chmod +x train/CIFAR100_train.sh
./train/CIFAR100_train.sh
```

- To produce trained models for different settings of the mini-ImageNet data; please run the following command.
```
chmod +x train/mini_train.sh
./train/mini_train.sh
```

- To produce trained models for different settings of the Food-101 data; please run the following command.
```
chmod +x train/food_train.sh
./train/food_train.sh
```

- To produce results for the eurosat data; please run the following command.
```
chmod +x test/test_aps2.sh
./test/test_aps2.sh
```