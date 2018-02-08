#!/usr/bin/env bash
python test.py -r checkpoints/binbranch_512/250_model.pkl
python test.py -r checkpoints/binbranch_512/200_model.pkl
python test.py -r checkpoints/binbranch_512/300_model.pkl
python test.py -r checkpoints/binbranch_512/200_model.pkl -test 0
