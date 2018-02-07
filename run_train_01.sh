#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss softneig  -epochs 601 -log_dir all_pos_28_neg_4  -save_step 50
python test.py -r checkpoints/all_pos_28_neg_4/model.pkl > all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/50_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/100_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/200_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/300_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/350_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/350_model.pkl -test 0 >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/400_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/450_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/500_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/550_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/600_model.pkl >> all_pos_28_neg_4.txt
python test.py -r checkpoints/all_pos_28_neg_4/600_model.pkl -test 0


