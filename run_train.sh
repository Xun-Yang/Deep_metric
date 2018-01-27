CUDA_VISIBLE_DEVICES=5 python train.py -data cub -net bn  -lr 1e-6 -dim 512 -num_instances 8 -BatchSize 128  -loss neighbour -epochs 401 -log_dir cub_m_01_1e6_n_8_b_128  -save_step 100

