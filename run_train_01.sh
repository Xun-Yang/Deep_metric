#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 python train.py -data cub -net bn -nums 0,52,102,152,204 -base 0.1  -init rand   -s 270  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss edwneig  -epochs 701 -log_dir edw_neighbour_unbalance  -save_step 50
python test.py -r checkpoints/edw_neighbour_unbalance/model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/50_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/100_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/200_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/300_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/350_model.pkl -test 0 >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/400_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/450_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/500_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/550_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/600_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/650_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/700_model.pkl >> result_bdwneig_esemble.txt
python test.py -r checkpoints/edw_neighbour_unbalance/700_model.pkl -test 0 >> result_bdwneig_esemble.txt


