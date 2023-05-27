#!/bin/bash
severity=(1 2 3 4)
gce_model_path = ('gce_severtity1','gce_severtity2','gce_severtity3','gce_severtity4')
gamma=(1 2 4 8)

for i in ${severity[@]}
do
    for j in ${gamma[@]}
    do
        python train_ours.py with server_user corrupted_cifar10 type0 skewed1 'severity'$i 'gamma'$j 'gce_model_path'$i
    done
done