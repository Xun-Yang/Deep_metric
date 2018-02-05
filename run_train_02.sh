#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -m 1 -net branch -base 0.1  -init rand  -s 200 -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss bdwneig  -epochs 601 -log_dir bdw_neighbour  -save_step 50
python test.py -r checkpoints/bdw_neighbour/model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/50_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/100_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/200_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/250_model.pkl -test 0 >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/300_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/400_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/500_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl -test 0 >> result_bdw.txt



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -m 1.0 -net branch -base 0.3  -init rand  -s 200 -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss bdwneig  -epochs 601 -log_dir bdw_neighbour  -save_step 50
python test.py -r checkpoints/bdw_neighbour/model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/50_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/100_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/200_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/250_model.pkl -test 0 >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/300_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/400_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/500_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl -test 0 >> result_bdw.txt



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -m 1.0 -net branch -base 0.5  -init rand  -s 200 -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss bdwneig  -epochs 601 -log_dir bdw_neighbour  -save_step 50
python test.py -r checkpoints/bdw_neighbour/model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/50_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/100_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/200_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/250_model.pkl -test 0 >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/300_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/400_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/500_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl -test 0 >> result_bdw.txt


#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -m 1.0 -net branch -base 1  -init rand  -s 200 -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss bdwneig  -epochs 601 -log_dir bdw_neighbour  -save_step 50
python test.py -r checkpoints/bdw_neighbour/model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/50_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/100_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/200_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/250_model.pkl -test 0 >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/300_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/400_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/500_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl -test 0 >> result_bdw.txt



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -m 1.0 -net branch -base 5 -init rand  -s 200 -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss bdwneig  -epochs 601 -log_dir bdw_neighbour  -save_step 50
python test.py -r checkpoints/bdw_neighbour/model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/50_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/100_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/200_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/250_model.pkl -test 0 >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/300_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/400_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/500_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl >> result_bdw.txt
python test.py -r checkpoints/bdw_neighbour/600_model.pkl -test 0 >> result_bdw.txt

