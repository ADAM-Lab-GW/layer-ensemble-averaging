Descriptions of provided code files:

- `dataset.py` - Defines the multi-task variant of the Yin-Yang dataset. Apapted from code by original authors.
- `train_multitask_ewc.py` - A PyTorch implementation of neural network training for the investigated network architecture, multi-task classification task with elastic weight consolidation.
- `layer_ensemble_averaging.py` - Defines helper functions for the core layer ensemble averaging framework and hardware neural network deployment for inference. Requires primitives from daffodil-lib and daffodil-app (will be publicly released soon).

NOTE: Code for quantizing software solutions is not included as it is based on the publicly available BRECQ implementation: https://github.com/yhhhli/BRECQ