#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python train.py -data cub -net bn  -lr 7e-7 -dim 512   -num_instances 8 -BatchSize 128  -loss positive  -epochs 601 -log_dir positive  -save_step 50
python test.py -r checkpoints/positive/model.pkl
python test.py -r checkpoints/positive/50_model.pkl
python test.py -r checkpoints/positive/100_model.pkl
python test.py -r checkpoints/positive/150_model.pkl
python test.py -r checkpoints/positive/200_model.pkl
python test.py -r checkpoints/positive/250_model.pkl
python test.py -r checkpoints/positive/300_model.pkl
python test.py -r checkpoints/positive/350_model.pkl
python test.py -r checkpoints/positive/400_model.pkl
python test.py -r checkpoints/positive/450_model.pkl
python test.py -r checkpoints/positive/500_model.pkl
python test.py -r checkpoints/positive/550_model.pkl
python test.py -r checkpoints/positive/600_model.pkl
python test.py -r checkpoints/positive/600_model.pkl -test 0
python test.py -r checkpoints/positive/350_model.pkl -test 0
