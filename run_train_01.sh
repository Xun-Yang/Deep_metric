#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python train.py -data cub -net bn  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 401 -log_dir dwdev  -save_step 50
python test.py -r checkpoints/dwdev/model.pkl
python test.py -r checkpoints/dwdev/50_model.pkl
python test.py -r checkpoints/dwdev/100_model.pkl
python test.py -r checkpoints/dwdev/150_model.pkl
python test.py -r checkpoints/dwdev/200_model.pkl
python test.py -r checkpoints/dwdev/250_model.pkl -test 0
python test.py -r checkpoints/dwdev/250_model.pkl
python test.py -r checkpoints/dwdev/300_model.pkl
python test.py -r checkpoints/dwdev/350_model.pkl
python test.py -r checkpoints/dwdev/400_model.pkl
python test.py -r checkpoints/dwdev/400_model.pkl -test 0
