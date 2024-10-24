#!/bin/bash

# get the sh file executable  chmod +x run.sh, run the sh file using nohup bash scripts/corruptedcifar10.sh &


seeds=(274 87 5 5881 1)

for seed in "${seeds[@]}"; do 
    echo "Running for seed ${seed}"
    CUDA_VISIBLE_DEVICES=0 python3 -u main.py with random_seed=${seed} bffhq; 
done



