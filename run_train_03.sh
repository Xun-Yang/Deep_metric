#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 python train.py -data cub -net bn  -base 0.1  -init rand   -s 150  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss distance_match  -epochs 601 -log_dir DM_3pos_10logexp  -save_step 50
python test.py -r checkpoints/DM_3pos_10logexp/model.pkl > result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/50_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/100_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/200_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/300_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/350_model.pkl -test 0 >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/400_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/450_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/500_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/550_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/600_model.pkl >> result_3pos_DM_3pos_10logexp.txt
python test.py -r checkpoints/DM_3pos_10logexp/600_model.pkl -test 0


