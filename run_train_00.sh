#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss all_pos_28_neg_2  -epochs 601 -log_dir all_pos_28_neg_2  -save_step 50
python test.py -r checkpoints/all_pos_28_neg_2/model.pkl > all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/50_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/100_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/200_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/300_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/350_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/350_model.pkl -test 0 >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/400_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/450_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/500_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/550_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/600_model.pkl >> all_pos_28_neg_2.txt
python test.py -r checkpoints/all_pos_28_neg_2/600_model.pkl -test 0


