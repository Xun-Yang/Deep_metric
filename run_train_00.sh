#Experiments 1
#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="64"
DATA="product"
loss="triplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=7 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM  -num_instances 4 -BatchSize 128 -loss $loss  -epochs 101 -log_dir $loss/$DATA/$DIM  -save_step 10

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="10 20 30 40 50 60 70 80 90 100"
for i in $Model_LIST; do
    python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done





# Experiments 2
#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="256"
DATA="product"
loss="triplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=7 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM  -num_instances 4 -BatchSize 128 -loss $loss  -epochs 101 -log_dir $loss/$DATA/$DIM  -save_step 10

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="10 20 30 40 50 60 70 80 90 100"
for i in $Model_LIST; do
    python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done



# Experiments 2
#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="1024"
DATA="product"
loss="triplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=7 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM  -num_instances 4 -BatchSize 128 -loss $loss  -epochs 101 -log_dir $loss/$DATA/$DIM  -save_step 10

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="10 20 30 40 50 60 70 80 90 100"
for i in $Model_LIST; do
    python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done


#Experiments 1
#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="64"
DATA="product"
loss="triplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=7 python triplet_train.py -data $DATA  -net bn  -init orth -lr 1e-5 -dim $DIM  -num_instances 4 -BatchSize 128 -loss $loss  -epochs 101 -log_dir $loss/$DATA/$DIM  -save_step 10

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="10 20 30 40 50 60 70 80 90 100"
for i in $Model_LIST; do
    python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done





# Experiments 3
#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="256"
DATA="product"
loss="Atriplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=7 python triplet_train.py -data $DATA  -alpha 40 -net bn  -init orth -lr 1e-5 -dim $DIM  -num_instances 4 -BatchSize 128 -loss $loss  -epochs 101 -log_dir $loss/$DATA/$DIM  -save_step 10

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="10 20 30 40 50 60 70 80 90 100"
for i in $Model_LIST; do
    python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done



# Experiments 2
#!/usr/bin/env bash
# use one GPU for on night is enough
DIM="1024"
DATA="product"
loss="Atriplet"
l="checkpoints/"$loss/$DATA/$DIM/
r="_model.pkl"

mkdir checkpoints/
mkdir checkpoints/$loss/
mkdir checkpoints/$loss/$DATA/
mkdir checkpoints/$loss/$DATA/$DIM

CUDA_VISIBLE_DEVICES=7 python triplet_train.py -data $DATA  -net bn -alpha 40 -init orth -lr 1e-5 -dim $DIM  -num_instances 4 -BatchSize 128 -loss $loss  -epochs 101 -log_dir $loss/$DATA/$DIM  -save_step 10

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

Model_LIST="10 20 30 40 50 60 70 80 90 100"
for i in $Model_LIST; do
    python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
done
