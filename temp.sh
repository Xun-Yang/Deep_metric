#!/usr/bin/env bash
DATA="car"
loss="mca"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"
DIM='512'
l=$checkpoints/$loss/$DATA/$DIM

Model_LIST="20"
for i in $Model_LIST; do
       CUDA_VISIBLE_DEVICES=1 python test.py -data $DATA -r $l/$i$r
done
