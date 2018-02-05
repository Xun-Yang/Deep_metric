#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn -base 0.1  -m 0.43 -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 601 -log_dir dwdev  -save_step 50
python test.py -r checkpoints/dwdev/model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/50_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/100_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/200_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/250_model.pkl -test 0 >> result_dev.txt
python test.py -r checkpoints/dwdev/300_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/400_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/500_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl -test 0 >> result_dev.txt



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn -base 0.3 -m 0.3  -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 601 -log_dir dwdev  -save_step 50
python test.py -r checkpoints/dwdev/model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/50_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/100_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/200_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/250_model.pkl -test 0 >> result_dev.txt
python test.py -r checkpoints/dwdev/300_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/400_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/500_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl -test 0 >> result_dev.txt



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn -base 0.5  -m 0.43  -init rand  -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 601 -log_dir dwdev  -save_step 50
python test.py -r checkpoints/dwdev/model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/50_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/100_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/200_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/250_model.pkl -test 0 >> result_dev.txt
python test.py -r checkpoints/dwdev/300_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/400_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/500_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl -test 0 >> result_dev.txt


#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn -base 1  -init rand  -m 0.43 -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 601 -log_dir dwdev  -save_step 50
python test.py -r checkpoints/dwdev/model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/50_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/100_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/200_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/250_model.pkl -test 0 >> result_dev.txt
python test.py -r checkpoints/dwdev/300_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/400_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/500_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl -test 0 >> result_dev.txt



#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn -base 5 -init rand  -m 0.43 -s 250  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdev  -epochs 601 -log_dir dwdev  -save_step 50
python test.py -r checkpoints/dwdev/model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/50_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/100_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/200_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/250_model.pkl -test 0 >> result_dev.txt
python test.py -r checkpoints/dwdev/300_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/400_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/500_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl >> result_dev.txt
python test.py -r checkpoints/dwdev/600_model.pkl -test 0 >> result_dev.txt

