#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -base 0.1  -init rand   -s 270  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss edwneig  -epochs 701 -log_dir edw_neighbour_128_4  -save_step 50
python test.py -r checkpoints/edw_neighbour_128_4/model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/50_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/100_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/200_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/300_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/350_model.pkl -test 0 >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/400_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/450_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/500_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/550_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/600_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/650_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/700_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_128_4/700_model.pkl -test 0 >> result_bdwneig_esemble.txt


