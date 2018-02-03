CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net branch  -lr 7e-7 -dim 512   -num_instances 8 -BatchSize 128  -loss binbranch  -epochs 601 -log_dir binbranch  -save_step 50
python test.py -r checkpoints/binbranch/model.pkl
python test.py -r checkpoints/binbranch/50_model.pkl
python test.py -r checkpoints/binbranch/100_model.pkl
python test.py -r checkpoints/binbranch/150_model.pkl
python test.py -r checkpoints/binbranch/200_model.pkl
python test.py -r checkpoints/binbranch/250_model.pkl
python test.py -r checkpoints/binbranch/300_model.pkl
python test.py -r checkpoints/binbranch/350_model.pkl
python test.py -r checkpoints/binbranch/400_model.pkl
python test.py -r checkpoints/binbranch/450_model.pkl
python test.py -r checkpoints/binbranch/500_model.pkl
python test.py -r checkpoints/binbranch/550_model.pkl
python test.py -r checkpoints/binbranch/600_model.pkl
python test.py -r checkpoints/binbranch/600_model.pkl -test 0
python test.py -r checkpoints/binbranch/350_model.pkl -test 0
