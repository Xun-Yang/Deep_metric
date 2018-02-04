#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python train.py -data cub -net branch  -m 0.42 -init rand -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss dwdevbranch  -epochs 201 -log_dir branch_dwdev  -save_step 50 -s 1
python test.py -r che4kpoints/branch_dwdev/model.pkl
python test.py -r checkpoints/branch_dwdev/50_model.pkl
python test.py -r checkpoints/branch_dwdev/100_model.pkl
python test.py -r checkpoints/branch_dwdev/150_model.pkl
python test.py -r checkpoints/branch_dwdev/200_model.pkl
python test.py -r checkpoints/branch_dwdev/250_model.pkl -test 0
python test.py -r checkpoints/branch_dwdev/250_model.pkl
python test.py -r checkpoints/branch_dwdev/300_model.pkl
python test.py -r checkpoints/branch_dwdev/350_model.pkl
python test.py -r checkpoints/branch_dwdev/400_model.pkl
python test.py -r checkpoints/branch_dwdev/400_model.pkl -test 0
