<<<<<<< HEAD
#!/usr/neighbour_rand/env bash
CUDA_VISIBLE_DEVICES=4 python train.py -data car -net bn  -base 1  -init rand  -s 200  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss neighbour_rand   -epochs 601 -log_dir neighbour_rand  -save_step 50
=======
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py -data cub -net bn  -base 0.1  -init rand   -s 200  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss bdwneig  -epochs 601 -log_dir neig  -save_step 50
python test.py -r checkpoints/neig/model.pkl  >neig.txt
python test.py -r checkpoints/neig/50_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/100_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/200_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/300_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/350_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/350_model.pkl  >>neig.txt-test 0 
python test.py -r checkpoints/neig/400_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/450_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/500_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/550_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/600_model.pkl  >>neig.txt
python test.py -r checkpoints/neig/600_model.pkl  >>neig.txt-test 0


>>>>>>> 73e147ea3b87714b28172fc609477482b58aab94
