#!/usr/neighbour_rand/env bash
CUDA_VISIBLE_DEVICES=4 python train.py -data car -net bn  -base 1  -init rand  -s 200  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss neighbour_rand   -epochs 601 -log_dir neighbour_rand  -save_step 50
