#!/usr/bin/env bash
DATA="cub"
loss="cluster-nca"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

DIM_list="512 48 64 96 128 256 384 1024"
for DIM in $DIM_list;do
    l=$checkpoints/$loss/$DATA/$DIM
    mkdir $checkpoints/$loss/$DATA/$DIM
    CUDA_VISIBLE_DEVICES=7 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 16 -beta 0.1 -n_cluster 25   -num_instances 8 -BatchSize 128 -loss $loss  -epochs 801 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM  -save_step 100
    Model_LIST="100 200 300 400 500 600 700 800"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
    done
done