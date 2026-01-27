#!/bin/bash

#SBATCH --partition=i64m1tga40u

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

# unset CUDA_VISIBLE_DEVICES




python main.py data=zsre \
    model=llama2-7b \
    editor=malmen \
    editor.lr=1e-5 \
