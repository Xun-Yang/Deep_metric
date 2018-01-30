#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-5 -dim  256   -num_instances 8 -BatchSize 128  -loss bin -epochs 801 -log_dir bin_256  -save_step 50
