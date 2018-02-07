#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss softneig -epochs 601 -log_dir softneig_18neg_3pos  -save_step 50
python test.py -r checkpoints/softneig_18neg_3pos/model.pkl > softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/50_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/100_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/200_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/300_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/350_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/350_model.pkl -test 0 >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/400_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/450_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/500_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/550_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/600_model.pkl >> softneig_18neg_3pos.txt
python test.py -r checkpoints/softneig_18neg_3pos/600_model.pkl -test 0


