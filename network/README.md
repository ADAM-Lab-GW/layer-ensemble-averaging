Descriptions of provided code files:

- `utils.py` - Defines the multi-task variant of the Yin-Yang dataset (adapted from code by original authors), as well as helper functions for network training and testing.
- `models.py` - Defines the neural network architecture.
- `train_multitask_ewc.py` - A PyTorch implementation of training the network for multi-task Yin-Yang classification task with/without elastic weight consolidation.
- `layer_ensemble_averaging.py` - A reference implementation of layer ensemble averaging. Requires primitives from `daffodil-lib` and `daffodil-app`. The code for interfacing with these classes from the custom mixed-signal prototyping platform will be made available by NIST as part of a future public release.

NOTE: Code for quantizing software solutions is not included as it is based on the publicly available BRECQ implementation: https://github.com/yhhhli/BRECQ