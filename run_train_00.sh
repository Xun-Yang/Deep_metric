CUDA_VISIBLE_DEVICES=5 python train.py -data cub -net bn  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss bin  -epochs 401 -log_dir mean_512  -save_step 100

