#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 python train.py -data cub -net bn  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss neighbour  -epochs 401 -log_dir neighbour  -save_step 50
python test.py -r checkpoints/neighbour/model.pkl
python test.py -r checkpoints/neighbour/50_model.pkl
python test.py -r checkpoints/neighbour/100_model.pkl
python test.py -r checkpoints/neighbour/150_model.pkl
python test.py -r checkpoints/neighbour/200_model.pkl
python test.py -r checkpoints/neighbour/250_model.pkl -test 0
python test.py -r checkpoints/neighbour/250_model.pkl
python test.py -r checkpoints/neighbour/300_model.pkl
python test.py -r checkpoints/neighbour/400_model.pkl
python test.py -r checkpoints/neighbour/400_model.pkl -test 0
