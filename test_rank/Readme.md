# MoE-PINNs: Mixture-of-Experts for Partial Differential Equations

This repository investigates the performance and weight dynamics (specifically the numerical rank) of Mixture-of-Experts (MoE) architectures when solving Partial Differential Equations (PDEs) using Physics-Informed Neural Networks (PINNs).

## 🚀 Quick Start

```bash
# train
python main.py --eq poisson --model moe_d --iters 1000

```

---

## ⚙️ Configuration Parameters

### 1. General Settings

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--eq` | `str` | `poisson` | Equation name (e.g., poisson,ac). |
| `--device` | `str` | `cuda:5` | Target device for computation. |
| `--model` | `str` | `moe_d` | Model architecture (`moe_d`, `mlp`, etc.). |
| `--seed` | `int` | `2026` | Random seed for reproducibility. |
| `--save_dir` | `str` | `...` | Path to save logs, checkpoints, and plots. |


### 3. Model 

| Parameter | Type | Default | Description |
| `--use_double` | `bool` | `False` | Use `float64` precision for training. |

### 4. Optimization Strategy

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--iters` | `int` | `1000` | Number of iterations for the Adam optimizer. |
| `--lr` | `float` | `8e-3` | Learning rate for Adam. |
| `--use_lbfgs` | `bool` | `False` | Enable L-BFGS optimization after Adam. |
| `--lbfgs_iter` | `int` | `500` | Maximum iterations for L-BFGS. |
| `--lbfgs_lr` | `float` | `1.0` | Learning rate for L-BFGS. |
| `--use_scheduler` | `bool` | `True` | Enable StepLR learning rate decay. |

### 5. Callbacks & Monitoring

The system uses a flexible callback frequency dictionary for `adam` and `lbfgs` phases:

* **`--log_freq`**: Frequency of printing loss to the console.
* **`--checkpoint_freq`**: Frequency of saving `.pt` model weights.
* **`--plot_freq`**: Frequency of generating error plots and solution visualizations.
* **`--rank_freq`**: Frequency of analyzing the numerical rank of expert weights.
* **`--loss_freq`**: Frequency of internal loss logging for the checkpoint system.

---

## 📊 Rank Analysis

A core feature of this repository is monitoring the **Numerical Rank** of expert matrices to detect "Expert Collapse" or representation redundancy.

* **`--eps`**: Threshold for singular value truncation when calculating rank.
---

## 📂 Output Structure

```text
save_dir/
├── _log_model/       # Saved checkpoints (*.pt and last.pt)
└── (logs)            # Training logs 
```

```

```