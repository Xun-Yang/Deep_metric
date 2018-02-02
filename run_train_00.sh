#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss margin  -epochs 601 -log_dir margin  -save_step 50
python test.py -r checkpoints/margin/model.pkl
python test.py -r checkpoints/margin/50_model.pkl
python test.py -r checkpoints/margin/100_model.pkl
python test.py -r checkpoints/margin/150_model.pkl
python test.py -r checkpoints/margin/200_model.pkl
python test.py -r checkpoints/margin/250_model.pkl
python test.py -r checkpoints/margin/300_model.pkl
python test.py -r checkpoints/margin/350_model.pkl
python test.py -r checkpoints/margin/400_model.pkl
python test.py -r checkpoints/margin/450_model.pkl
python test.py -r checkpoints/margin/500_model.pkl
python test.py -r checkpoints/margin/550_model.pkl
python test.py -r checkpoints/margin/600_model.pkl
python test.py -r checkpoints/margin/600_model.pkl -test 0
python test.py -r checkpoints/margin/350_model.pkl -test 0
