#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss softneig  -epochs 601 -log_dir  6_pos_18_neg3_alpha20  -save_step 50
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/model.pkl
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/50_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/100_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/200_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/300_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/350_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/350_model.pkl -test 0 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/400_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/450_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/500_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/550_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/600_model.pkl 
python test.py -r checkpoints/ 6_pos_18_neg3_alpha20/600_model.pkl -test 0


