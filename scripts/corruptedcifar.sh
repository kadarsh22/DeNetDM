#!/bin/bash

# get the sh file executable  chmod +x run.sh, run the sh file using chmod +x run.sh


seeds=(274 87 5 5881 67)

for seed in "${seeds[@]}"; do 
    echo "Running for seed ${seed}"
    python3 -u main.py with random_seed=${seed} corrupted_cifar10 type0 skewed3 severity4; 
done