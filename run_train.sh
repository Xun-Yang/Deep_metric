#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-6 -dim  512   -num_instances 8 -BatchSize 128  -loss bin -epochs 401 -log_dir pos_bin_512  -save_step 50
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-6 -dim  256   -num_instances 8 -BatchSize 128  -loss bin -epochs 401 -log_dir pos_bin_256  -save_step 50
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-6 -dim  128   -num_instances 8 -BatchSize 128  -loss bin -epochs 401 -log_dir pos_bin_128  -save_step 50
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-6 -dim  64   -num_instances 8 -BatchSize 128  -loss bin -epochs 601 -log_dir pos_bin_64  -save_step 50
