#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdevbranch  -epochs 401 -log_dir dwdevbranch  -save_step 50
python test.py -r checkpoints/dwdevbranch/model.pkl
python test.py -r checkpoints/dwdevbranch/50_model.pkl
python test.py -r checkpoints/dwdevbranch/100_model.pkl
python test.py -r checkpoints/dwdevbranch/150_model.pkl
python test.py -r checkpoints/dwdevbranch/200_model.pkl
python test.py -r checkpoints/dwdevbranch/250_model.pkl -test 0
python test.py -r checkpoints/dwdevbranch/250_model.pkl
python test.py -r checkpoints/dwdevbranch/300_model.pkl
python test.py -r checkpoints/dwdevbranch/350_model.pkl
python test.py -r checkpoints/dwdevbranch/400_model.pkl
python test.py -r checkpoints/dwdevbranch/400_model.pkl -test 0
