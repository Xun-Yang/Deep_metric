#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -base 0.1  -init rand   -s 150  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss distance_match  -epochs 601 -log_dir distance_match  -save_step 50
python test.py -r checkpoints/distance_match/model.pkl > result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/50_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/100_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/200_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/300_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/350_model.pkl -test 0 >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/400_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/450_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/500_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/550_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/600_model.pkl >> result_all_pos_DM.txt
python test.py -r checkpoints/distance_match/600_model.pkl -test 0 


