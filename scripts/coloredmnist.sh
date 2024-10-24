#!/bin/bash

# get the sh file executable chmod +x run.sh, run the sh file using nohup bash scripts/coloredmnist.sh &

# seeds=(274 87 5 5881 1)
# skew=(skewed1 skewed2 skewed3 skewed4)

seeds=(1 2 3 4 5)
skew=(skewed1)

for skewness in "${skew[@]}"; do
    for seed in "${seeds[@]}"; do 
        echo "Running for seed number ${seed}"
        CUDA_VISIBLE_DEVICES=0 python3 -u main.py with colored_mnist $skewness severity4;
    done
done

