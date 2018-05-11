#!/usr/bin/env bash
DATA="cub"
loss="mca"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

DIM_list="512 64"
for DIM in $DIM_list;do
    l=$checkpoints/$loss/$DATA/$DIM
    mkdir $checkpoints/$loss/$DATA/$DIM
    CUDA_VISIBLE_DEVICES=7   python MCA_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 24 -n_cluster 2 -center_init random  -BatchSize 128 -loss $loss  -epochs 51 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM  -save_step 10
    Model_LIST="10 20 30 40 50"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
    done
done
