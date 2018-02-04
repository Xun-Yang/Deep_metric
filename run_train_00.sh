#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -base 0.1  -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwneig  -epochs 601 -log_dir dw_neighbour  -save_step 50
python test.py -r checkpoints/dw_neighbour/model.pkl
python test.py -r checkpoints/dw_neighbour/50_model.pkl
python test.py -r checkpoints/dw_neighbour/100_model.pkl
python test.py -r checkpoints/dw_neighbour/200_model.pkl
python test.py -r checkpoints/dw_neighbour/250_model.pkl -test 0
python test.py -r checkpoints/dw_neighbour/300_model.pkl
python test.py -r checkpoints/dw_neighbour/400_model.pkl
python test.py -r checkpoints/dw_neighbour/500_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl -test 0



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -base 0.3  -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwneig  -epochs 601 -log_dir dw_neighbour  -save_step 50
python test.py -r checkpoints/dw_neighbour/model.pkl
python test.py -r checkpoints/dw_neighbour/50_model.pkl
python test.py -r checkpoints/dw_neighbour/100_model.pkl
python test.py -r checkpoints/dw_neighbour/200_model.pkl
python test.py -r checkpoints/dw_neighbour/250_model.pkl -test 0
python test.py -r checkpoints/dw_neighbour/300_model.pkl
python test.py -r checkpoints/dw_neighbour/400_model.pkl
python test.py -r checkpoints/dw_neighbour/500_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl -test 0



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -base 0.5  -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwneig  -epochs 601 -log_dir dw_neighbour  -save_step 50
python test.py -r checkpoints/dw_neighbour/model.pkl
python test.py -r checkpoints/dw_neighbour/50_model.pkl
python test.py -r checkpoints/dw_neighbour/100_model.pkl
python test.py -r checkpoints/dw_neighbour/200_model.pkl
python test.py -r checkpoints/dw_neighbour/250_model.pkl -test 0
python test.py -r checkpoints/dw_neighbour/300_model.pkl
python test.py -r checkpoints/dw_neighbour/400_model.pkl
python test.py -r checkpoints/dw_neighbour/500_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl -test 0


#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -base 1  -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwneig  -epochs 601 -log_dir dw_neighbour  -save_step 50
python test.py -r checkpoints/dw_neighbour/model.pkl
python test.py -r checkpoints/dw_neighbour/50_model.pkl
python test.py -r checkpoints/dw_neighbour/100_model.pkl
python test.py -r checkpoints/dw_neighbour/200_model.pkl
python test.py -r checkpoints/dw_neighbour/250_model.pkl -test 0
python test.py -r checkpoints/dw_neighbour/300_model.pkl
python test.py -r checkpoints/dw_neighbour/400_model.pkl
python test.py -r checkpoints/dw_neighbour/500_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl -test 0



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn -base 5 -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwneig  -epochs 601 -log_dir dw_neighbour  -save_step 50
python test.py -r checkpoints/dw_neighbour/model.pkl
python test.py -r checkpoints/dw_neighbour/50_model.pkl
python test.py -r checkpoints/dw_neighbour/100_model.pkl
python test.py -r checkpoints/dw_neighbour/200_model.pkl
python test.py -r checkpoints/dw_neighbour/250_model.pkl -test 0
python test.py -r checkpoints/dw_neighbour/300_model.pkl
python test.py -r checkpoints/dw_neighbour/400_model.pkl
python test.py -r checkpoints/dw_neighbour/500_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl
python test.py -r checkpoints/dw_neighbour/600_model.pkl -test 0

