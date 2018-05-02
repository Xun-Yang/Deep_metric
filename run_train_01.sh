#!/usr/bin/env bash
# use one GPU for on night is enough
DATA="cub"
loss="neighbour"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

k_list="1 2 4 6 8"
margin="0.1"
for k in $k_list;do
    l=$checkpoints/$loss/$DATA/$k_$margin
    mkdir $checkpoints/$loss/$DATA/$k_$margin
    CUDA_VISIBLE_DEVICES=1 python train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim 64 -num_instances 8 -BatchSize 128  -loss $loss  -k $k -margin $margin -epochs 801 -checkpoints $checkpoints -log_dir $loss/$DATA/$k_$margin  -save_step 100
    Model_LIST="100 200 300 400 500 600 700 800"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=1  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$k_$margin.txt
    done
done

