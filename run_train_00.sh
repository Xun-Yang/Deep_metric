#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python train.py -data cub -net bn -base 0.1  -init rand   -s 270  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss edwneig  -epochs 701 -log_dir edw_neighbour  -save_step 50
python test.py -r checkpoints/edw_neighbour/model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/50_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/100_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/200_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/300_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/350_model.pkl -test 0 >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/400_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/450_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/500_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/550_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/600_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/650_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/700_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour/700_model.pkl -test 0 >> result_bdwneig_esemble.txt


