#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -alpha 40  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax  -save_step 50
python test.py -r checkpoints/knnsoftmax/model.pkl  >knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/50_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/100_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/200_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/300_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/350_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/350_model.pkl -test 0 >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/400_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/450_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/500_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/550_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/600_model.pkl  >>knnsoftmax.txt
python test.py -r checkpoints/knnsoftmax/600_model.pkl -test 0 >>knnsoftmax.txt


