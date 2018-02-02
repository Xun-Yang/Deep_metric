CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss margin  -epochs 401 -log_dir margin  -save_step 50

