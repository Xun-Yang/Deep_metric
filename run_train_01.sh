CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -lr 1e-5 -dim 128  -num_instances 8 -BatchSize 128  -loss neighbour -epochs 801 -log_dir 128_dim  -save_step 100
CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -lr 1e-5 -dim 256   -num_instances 8 -BatchSize 128  -loss neighbour -epochs 801 -log_dir 256_dim  -save_step 100

