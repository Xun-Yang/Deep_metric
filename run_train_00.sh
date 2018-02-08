#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss softneig  -epochs 601 -log_dir 6_pos_18_neg2_alpha50  -save_step 50
python test.py -r checkpoints/6_pos_18_neg2_alpha50/model.pkl  >6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/50_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/100_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/200_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/300_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/350_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/350_model.pkl  >>6_pos_18_neg2_alpha50.txt-test 0 
python test.py -r checkpoints/6_pos_18_neg2_alpha50/400_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/450_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/500_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/550_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/600_model.pkl  >>6_pos_18_neg2_alpha50.txt
python test.py -r checkpoints/6_pos_18_neg2_alpha50/600_model.pkl  >>6_pos_18_neg2_alpha50.txt-test 0


