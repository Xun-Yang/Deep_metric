#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss distance_match  -epochs 601 -log_dir dist_match_neig_mine  -save_step 50
python test.py -r checkpoints/dist_match_neig_mine/model.pkl > result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/50_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/100_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/200_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/300_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/350_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/350_model.pkl -test 0 >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/400_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/450_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/500_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/550_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/600_model.pkl >> result_3pos_dist_match_neig_mine.txt
python test.py -r checkpoints/dist_match_neig_mine/600_model.pkl -test 0


