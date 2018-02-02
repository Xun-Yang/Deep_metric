#!/usr/bin/env bash
python test.py -r checkpoints/binbranch_512/350_model.pkl
python test.py -r checkpoints/binbranch_512/250_model.pkl
python test.py -r checkpoints/binbranch_512/400_model.pkl
python test.py -r checkpoints/binbranch_512/450_model.pkl
python test.py -r checkpoints/binbranch_512/150_model.pkl
python test.py -r checkpoints/binbranch_512/100_model.pkl
python test.py -r checkpoints/binbranch_512/400_model.pkl -test 0
