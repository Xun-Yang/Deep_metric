#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python train.py -data cub -net bn  -lr 7e-7 -dim 128   -num_instances 8 -BatchSize 128  -loss dwcon  -epochs 401 -log_dir dwcon  -save_step 50
python test.py -r checkpoints/dwcon/model.pkl
python test.py -r checkpoints/dwcon/50_model.pkl
python test.py -r checkpoints/dwcon/100_model.pkl
python test.py -r checkpoints/dwcon/150_model.pkl
python test.py -r checkpoints/dwcon/200_model.pkl
python test.py -r checkpoints/dwcon/250_model.pkl
python test.py -r checkpoints/dwcon/300_model.pkl
python test.py -r checkpoints/dwcon/350_model.pkl
python test.py -r checkpoints/dwcon/400_model.pkl
python test.py -r checkpoints/dwcon/400_model.pkl -test 0
python test.py -r checkpoints/dwcon/250_model.pkl -test 0
