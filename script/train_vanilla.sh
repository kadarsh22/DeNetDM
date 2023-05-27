#!/bin/bash
severity=('severity1' 'severity2' 'severity3' 'severity4')

for i in ${severity[@]}
do
    python train_vanilla.py with server_user corrupted_cifar10 type0 skewed1 $i
done

severity=('severity1' 'severity2' 'severity3' 'severity4')

for i in ${severity[@]}
do
    python train.py with server_user corrupted_cifar10 type0 skewed1 $i
done
