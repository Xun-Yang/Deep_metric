#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -base 0.1  -init rand   -nums 0,102,102,102,102,104  -s 270  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss edwneig  -epochs 701 -log_dir edw_neighbour_102_5  -save_step 50
python test.py -r checkpoints/edw_neighbour_102_5/model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/50_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/100_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/200_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/300_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/350_model.pkl -test 0 >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/400_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/450_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/500_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/550_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/600_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/650_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/700_model.pkl >> result_102_5_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_102_5/700_model.pkl -test 0 >> result_102_5_bdwneig_esemble.txt


