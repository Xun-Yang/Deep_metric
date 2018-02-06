#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwneig  -epochs 601 -log_dir dwneig_2pos  -save_step 50
python test.py -r checkpoints/dwneig_2pos/model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/50_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/100_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/200_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/300_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/350_model.pkl -test 0 >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/400_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/450_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/500_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/550_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/600_model.pkl >> result_2pos_dwneig_2pos.txt
python test.py -r checkpoints/dwneig_2pos/600_model.pkl -test 0 >> result_2pos_dwneig_2pos.txt


