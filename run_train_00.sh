#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -m 0.4 -init norm -orth_cof 1e-6  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 401 -log_dir norm_dwdev  -save_step 50
python test.py -r checkpoints/norm_dwdev/model.pkl
python test.py -r checkpoints/norm_dwdev/50_model.pkl
python test.py -r checkpoints/norm_dwdev/100_model.pkl
python test.py -r checkpoints/norm_dwdev/150_model.pkl
python test.py -r checkpoints/norm_dwdev/200_model.pkl
python test.py -r checkpoints/norm_dwdev/250_model.pkl -test 0
python test.py -r checkpoints/norm_dwdev/250_model.pkl
python test.py -r checkpoints/norm_dwdev/300_model.pkl
python test.py -r checkpoints/norm_dwdev/350_model.pkl
python test.py -r checkpoints/norm_dwdev/400_model.pkl
python test.py -r checkpoints/norm_dwdev/400_model.pkl -test 0
