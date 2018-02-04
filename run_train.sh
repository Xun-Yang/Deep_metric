#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-6 -dim 128   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 801 -log_dir 128  -save_step 50
python test.py -r checkpoints/128/model.pkl
python test.py -r checkpoints/128/50_model.pkl
python test.py -r checkpoints/128/100_model.pkl
python test.py -r checkpoints/128/150_model.pkl
python test.py -r checkpoints/128/200_model.pkl
python test.py -r checkpoints/128/150_model.pkl -test 0
python test.py -r checkpoints/128/250_model.pkl
python test.py -r checkpoints/128/300_model.pkl
python test.py -r checkpoints/128/500_model.pkl
python test.py -r checkpoints/128/600_model.pkl
python test.py -r checkpoints/128/700_model.pkl
python test.py -r checkpoints/128/300_model.pkl
python test.py -r checkpoints/128/300_model.pkl -test 0
