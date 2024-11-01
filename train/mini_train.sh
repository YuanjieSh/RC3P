#!/bin/bash

python train/mini_train.py --gpu 0 --imb_type exp --imb_factor 0.5 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type exp --imb_factor 0.4 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type exp --imb_factor 0.3 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type exp --imb_factor 0.2 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type exp --imb_factor 0.1 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 

python train/mini_train.py --gpu 0 --imb_type poly --imb_factor 0.5 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type poly --imb_factor 0.4 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type poly --imb_factor 0.3 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type poly --imb_factor 0.2 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type poly --imb_factor 0.1 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 

python train/mini_train.py --gpu 0 --imb_type major --imb_factor 0.5 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type major --imb_factor 0.4 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type major --imb_factor 0.3 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type major --imb_factor 0.2 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/mini_train.py --gpu 0 --imb_type major --imb_factor 0.1 -b 128 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 


