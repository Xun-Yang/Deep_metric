#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-6 -dim 128   -num_instances 8 -BatchSize 128  -loss con  -epochs 401 -log_dir con  -save_step 50
python test.py -r checkpoints/con/model.pkl
python test.py -r checkpoints/con/50_model.pkl
python test.py -r checkpoints/con/100_model.pkl
python test.py -r checkpoints/con/150_model.pkl
python test.py -r checkpoints/con/200_model.pkl
python test.py -r checkpoints/con/250_model.pkl
python test.py -r checkpoints/con/250_model.pkl -test 0
python test.py -r checkpoints/con/300_model.pkl
python test.py -r checkpoints/con/350_model.pkl
python test.py -r checkpoints/con/400_model.pkl
python test.py -r checkpoints/con/350_model.pkl -test 0
