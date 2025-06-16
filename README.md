# Mixture-of-Experts (MoE)

This is a minimal PyTorch implementation of the Mixture-of-Experts (MoE) architecture. It demonstrates expert selection via a top-k gating mechanism with softmax normalization, sparse dispatching of inputs, and weighted aggregation of expert outputs. The model supports configuration of input size, number of experts, hidden layer size, loss function, and optimizer through command-line arguments.

The project consists of a main training script (`main.py`) and the core MoE components (`moe.py`), including the expert modules and gating logic.


Author: [@klaezhou](https://github.com/klaezhou)  
Date: June 2025
