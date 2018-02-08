#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_alpha30  -save_step 50
python test.py -r checkpoints/knnsoftmax_alpha30/model.pkl > knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/50_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/100_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/200_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/300_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/350_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/350_model.pkl -test 0 >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/400_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/450_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/500_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/550_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/600_model.pkl >> knnsoftmax_alpha30.txt
python test.py -r checkpoints/knnsoftmax_alpha30/600_model.pkl -test 0


