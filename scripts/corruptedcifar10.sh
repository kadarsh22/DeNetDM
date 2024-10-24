#!/bin/bash

# get the sh file executable  chmod +x run.sh, run the sh file using nohup bash scripts/corruptedcifar10.sh &

seeds=(274 87 5 5881 1)
skew=(skewed1 skewed2 skewed3 skewed4)

for skewness in "${skew[@]}"; do
    for seed in "${seeds[@]}"; do 
        echo "Running for seed ${seed}"
        CUDA_VISIBLE_DEVICES=1 python3 -u main.py with random_seed=${seed} corrupted_cifar10 type0 $skewness severity4;
    done
done

