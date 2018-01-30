#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py -data cub -net bn  -lr 1e-5 -dim  256   -num_instances 8 -BatchSize 128  -loss distweight -epochs 801 -log_dir dist_256_dim  -save_step 100
