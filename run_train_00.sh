CUDA_VISIBLE_DEVICES=5 python train.py -data cub -net branch  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss binbranch  -epochs 401 -log_dir pos_512  -save_step 50

