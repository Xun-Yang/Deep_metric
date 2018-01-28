CUDA_VISIBLE_DEVICES=6 python train.py -data cub -net bn  -lr 1e-5 -dim 1024   -num_instances 8 -BatchSize 128  -loss neighbour -epochs 801 -log_dir cub_1e5_n_8_b_128_mean  -save_step 100

