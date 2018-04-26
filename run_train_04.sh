#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="256"
DATA="car"
loss="Abatchall"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=8 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 40  -num_instances 8 -BatchSize 128 -loss $loss  -epochs 1001 -log_dir $loss/$DATA/$DIM  -save_step 100

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="100 200 300 400 500 600 700 800 900 1000"
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=8  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done






DIM="256"
DATA="car"
loss="batchall"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=8 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 40  -num_instances 8 -BatchSize 128 -loss $loss  -epochs 1001 -log_dir $loss/$DATA/$DIM  -save_step 100

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="100 200 300 400 500 600 700 800 900 1000"
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=8  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done





DIM="256"
DATA="car"
loss="triplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=8 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 4  -num_instances 8 -BatchSize 128 -loss $loss  -epochs 1001 -log_dir $loss/$DATA/$DIM  -save_step 100

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="100 200 300 400 500 600 700 800 900 1000"
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=8  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done




DIM="256"
DATA="car"
loss="Atriplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=8 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM -alpha 4  -num_instances 8 -BatchSize 128 -loss $loss  -epochs 1001 -log_dir $loss/$DATA/$DIM  -save_step 100

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="100 200 300 400 500 600 700 800 900 1000"
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=8  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done
