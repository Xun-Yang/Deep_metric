#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="64"
DATA="cub"
loss="nca"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"


mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=1 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 40   -num_instances 8 -BatchSize 128 -loss $loss  -epochs 1001 -log_dir $loss/$DATA/$DIM  -save_step 100


mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


