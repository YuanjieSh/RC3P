#!/bin/bash

python train/cifar100_train.py --gpu 1 --imb_type exp --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type exp --imb_factor 0.4 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type exp --imb_factor 0.3 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type exp --imb_factor 0.2 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100

python train/cifar100_train.py --gpu 1 --imb_type poly --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type poly --imb_factor 0.4 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type poly --imb_factor 0.3 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type poly --imb_factor 0.2 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type poly --imb_factor 0.1 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100

python train/cifar100_train.py --gpu 1 --imb_type major --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type major --imb_factor 0.4 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type major --imb_factor 0.3 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type major --imb_factor 0.2 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/cifar100_train.py --gpu 1 --imb_type major --imb_factor 0.1 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100

python train/plot_hist.py --gpu 0 --imb_type exp --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/plot_hist.py --gpu 0 --imb_type poly --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100
python train/plot_hist.py --gpu 0 --imb_type major --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 --dataset cifar100


