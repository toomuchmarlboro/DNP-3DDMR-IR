#!/bin/bash

for fold in 0 1 2 3 4; do
    echo "========================================="
    echo "Starting Fold $fold"
    echo "========================================="
    
    # Run the training script via torchrun
    torchrun --nproc_per_node=2 thermamnerf_v3.0.py --fold $fold
    
    if [ $? -ne 0 ]; then
        echo "Fold $fold failed! Exiting."
        exit 1
    fi
done

echo "========================================="
echo "ALL 5 FOLDS COMPLETED SUCCESSFULLY!"
echo "========================================="
