#!/usr/bin/env bash
# use one GPU for on night is enough
DATA="cub"
loss="nca"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

DIM_list="48 64 96 128 256 384 512 1024"
for DIM in $DIM_list;do
    l=$checkpoints/$loss/$DATA/$DIM
    mkdir $checkpoints/$loss/$DATA/$DIM
    CUDA_VISIBLE_DEVICES=1 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 16  -num_instances 8 -BatchSize 128 -loss $loss  -epochs 1001 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM  -save_step 100
    Model_LIST="100 200 300 400 500 600 700 800 900 1000"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=1  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
    done
done


