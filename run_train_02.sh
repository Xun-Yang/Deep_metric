#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn  -base 0.1  -init rand   -s 150  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss distance_match  -epochs 601 -log_dir DM3pos_used95base  -save_step 50
python test.py -r checkpoints/DM3pos_used95base/model.pkl > result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/50_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/100_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/200_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/300_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/350_model.pkl -test 0 >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/400_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/450_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/500_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/550_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/600_model.pkl >> result_3pos_DM3pos_used95base.txt
python test.py -r checkpoints/DM3pos_used95base/600_model.pkl -test 0


