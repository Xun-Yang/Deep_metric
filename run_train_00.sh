#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -init rand   -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dw_neighbour  -epochs 401 -log_dir dw_neighbour  -save_step 50
python test.py -r checkpoints/dw_neighbour/model.pkl
python test.py -r checkpoints/dw_neighbour/50_model.pkl
python test.py -r checkpoints/dw_neighbour/100_model.pkl
python test.py -r checkpoints/dw_neighbour/150_model.pkl
python test.py -r checkpoints/dw_neighbour/200_model.pkl
python test.py -r checkpoints/dw_neighbour/250_model.pkl -test 0
python test.py -r checkpoints/dw_neighbour/250_model.pkl
python test.py -r checkpoints/dw_neighbour/300_model.pkl
python test.py -r checkpoints/dw_neighbour/350_model.pkl
python test.py -r checkpoints/dw_neighbour/400_model.pkl
python test.py -r checkpoints/dw_neighbour/400_model.pkl -test 0
