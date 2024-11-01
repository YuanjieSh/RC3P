#!/bin/bash

python train/food_train.py --gpu 0 --imb_type exp -b 256 --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/food_train.py --gpu 0 --imb_type exp -b 256 --imb_factor 0.4 --loss_type LDAM --train_rule None -a resnet20 --epochs 200
python train/food_train.py --gpu 0 --imb_type exp -b 256 --imb_factor 0.3 --loss_type LDAM --train_rule None -a resnet20 --epochs 200
python train/food_train.py --gpu 0 --imb_type exp -b 256 --imb_factor 0.2 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/food_train.py --gpu 0 --imb_type exp -b 256 --imb_factor 0.1 --loss_type LDAM --train_rule None -a resnet20 --epochs 200

python train/food_train.py --gpu 0 --imb_type poly -b 256 --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/food_train.py --gpu 0 --imb_type poly -b 256 --imb_factor 0.4 --loss_type LDAM --train_rule None -a resnet20 --epochs 200
python train/food_train.py --gpu 0 --imb_type poly -b 256 --imb_factor 0.3 --loss_type LDAM --train_rule None -a resnet20 --epochs 200
python train/food_train.py --gpu 0 --imb_type poly -b 256 --imb_factor 0.2 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/food_train.py --gpu 0 --imb_type poly -b 256 --imb_factor 0.1 --loss_type LDAM --train_rule None -a resnet20 --epochs 200

python train/food_train.py --gpu 0 --imb_type major -b 256 --imb_factor 0.5 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/food_train.py --gpu 0 --imb_type major -b 256 --imb_factor 0.4 --loss_type LDAM --train_rule None -a resnet20 --epochs 200
python train/food_train.py --gpu 0 --imb_type major -b 256 --imb_factor 0.3 --loss_type LDAM --train_rule None -a resnet20 --epochs 200
python train/food_train.py --gpu 0 --imb_type major -b 256 --imb_factor 0.2 --loss_type LDAM --train_rule None -a resnet20 --epochs 200 
python train/food_train.py --gpu 0 --imb_type major -b 256 --imb_factor 0.1 --loss_type LDAM --train_rule None -a resnet20 --epochs 200
