#!/usr/bin/env bash
# use one GPU for on night is enough
DATA="car"
loss="neighbour"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

k="2"
margin_list="0.01 0.05 0.15 0.2"
for margin in $margin_list;do
    l=$checkpoints/$loss/$DATA/$k-$margin
    mkdir $checkpoints/$loss/$DATA/$k-$margin
    CUDA_VISIBLE_DEVICES=6 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim 64 -num_instances 8 -BatchSize 128  -loss $loss  -k $k -margin $margin -epochs 801 -checkpoints $checkpoints -log_dir $loss/$DATA/$k-$margin  -save_step 100
    Model_LIST="100 200 300 400 500 600 700 800"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=6  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$k-$margin.txt
    done
done

