#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="512"
DATA="cub"
loss="nca"
checkpoints="/opt/intern/users/xunwang/checkpoints"
l=$checkpoints/$loss/$DATA/$DIM
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/
mkdir $checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=2 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 16  -num_instances 8 -BatchSize 128 -loss $loss  -epochs 1001 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM  -save_step 100


mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="100 200 300 400 500 600 700 800 900 1000"
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=2  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done


