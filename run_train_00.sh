#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python train.py -data cub -net bn -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss neighbour  -epochs 701 -log_dir neighbour  -save_step 50
python test.py -r checkpoints/neighbour/model.pkl > neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/50_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/100_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/200_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/300_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/350_model.pkl -test 0 >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/400_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/450_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/500_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/550_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/600_model.pkl >> neighbour_adapative_margin.txt
python test.py -r checkpoints/neighbour/600_model.pkl -test 0 >> neighbour_adapative_margin.txt


