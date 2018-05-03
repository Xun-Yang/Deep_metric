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

DIM_list="512 64"
Beta_list='0.05 0.1 0.2 0.25 0.3'
for Beta in $Beta_list;do
for Dim in $DIM_list;do
    l=$checkpoints/$loss/$DATA/$Beta-$Dim
    mkdir $checkpoints/$loss/$DATA/$Beta-$Dim
    CUDA_VISIBLE_DEVICES=7 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $Dim -alpha 20 -beta $Beta -n_cluster 30   -num_instances 8 -BatchSize 128 -loss $loss  -epochs 801 -checkpoints $checkpoints -log_dir $loss/$DATA/$Beta-$Dim  -save_step 100
    Model_LIST="100 200 300 400 500 600 700 800"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $lc/$i$r >>result/$loss/$DATA/$Beta-$Dim.txt
    done
done
done
