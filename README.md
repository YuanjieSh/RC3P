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