#!/usr/binbranch/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net branch  -lr 1e-6 -dim  512   -num_instances 8 -BatchSize 128  -loss binbranch -epochs 601 -log_dir binbranch_512  -save_step 50
