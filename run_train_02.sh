#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss softneig -epochs 601 -log_dir softneig_28neg_all_pos  -save_step 50
python test.py -r checkpoints/softneig_28neg_all_pos/model.pkl > softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/50_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/100_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/200_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/300_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/350_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/350_model.pkl -test 0 >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/400_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/450_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/500_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/550_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/600_model.pkl >> softneig_28neg_all_pos.txt
python test.py -r checkpoints/softneig_28neg_all_pos/600_model.pkl -test 0


