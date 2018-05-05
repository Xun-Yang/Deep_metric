#!/usr/bin/env bash
# use one GPU for on night is enough
DATA="product"
loss="neighbour"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

margin_list="0.01 0.05 0.15 0.2"
k=1
for margin in $margin_list;do
    l=$checkpoints/$loss/$DATA/$k-$margin
    mkdir $checkpoints/$loss/$DATA/$k-$margin
    CUDA_VISIBLE_DEVICES=7 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim 128 -num_instances 4 -BatchSize 140  -loss $loss  -k $k -margin $margin -epochs 81 -checkpoints $checkpoints -log_dir $loss/$DATA/$k-$margin  -save_step 10
    Model_LIST="40 50 60 70 80"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$k-$margin.txt
    done
done

