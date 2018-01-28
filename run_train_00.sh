CUDA_VISIBLE_DEVICES=1 python train.py -data cub -net bn  -lr 1e-5 -dim 32   -num_instances 8 -BatchSize 128  -loss neighbour -epochs 801 -log_dir 32_dim  -save_step 100
CUDA_VISIBLE_DEVICES=1 python train.py -data cub -net bn  -lr 1e-5 -dim 64   -num_instances 8 -BatchSize 128  -loss neighbour -epochs 801 -log_dir 64_dim  -save_step 100

