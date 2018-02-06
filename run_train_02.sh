#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python train.py -data cub -net bn -base 0.1  -init rand   -nums 0,256,256  -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss edwneig  -epochs 601 -log_dir edw_neighbour_256_2  -save_step 50
python test.py -r checkpoints/edw_neighbour_256_2/model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/50_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/100_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/200_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/300_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/350_model.pkl -test 0 >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/400_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/450_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/500_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/550_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/600_model.pkl >> result_256_2_bdwneig_ada_margin_esemble.txt
python test.py -r checkpoints/edw_neighbour_256_2/600_model.pkl -test 0 >> result_256_2_bdwneig_ada_margin_esemble.txt


