#!/bin/bash
echo $HOSTNAME
export CUDA_VISIBLE_DEVICES=7
python run.py >> resultnew.txt 
