#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwneig  -epochs 601 -log_dir dwneig  -save_step 50
python test.py -r checkpoints/dwneig/model.pkl > result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/50_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/100_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/200_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/300_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/350_model.pkl -test 0 >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/400_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/450_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/500_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/550_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/600_model.pkl >> result_3pos_dwneig.txt
python test.py -r checkpoints/dwneig/600_model.pkl -test 0 


