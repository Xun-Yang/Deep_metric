#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss softneig  -epochs 601 -log_dir softneig  -save_step 50
python test.py -r checkpoints/softneig/model.pkl > softneig.txt
python test.py -r checkpoints/softneig/50_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/100_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/200_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/300_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/350_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/350_model.pkl -test 0 >> softneig.txt
python test.py -r checkpoints/softneig/400_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/450_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/500_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/550_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/600_model.pkl >> softneig.txt
python test.py -r checkpoints/softneig/600_model.pkl -test 0


