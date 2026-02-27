# 🚗 US Accidents Severity Prediction with Graph Neural Networks

**Group 8** — Graph Neural Network comparison study on the US Accidents dataset

## Overview

This project applies multiple Graph Neural Network (GNN) architectures to predict the severity of US traffic accidents, framed as a node classification problem. Each accident is represented as a node in a graph, with edges connecting geographically or state-proximate incidents. We compare four GNN models across multiple hyperparameter configurations to identify the best-performing approach.

## Dataset

**Source:** [US Accidents (March 2023) — Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

- ~7.7 million accident records spanning 2016–2023
- 46 features including location, weather, road features, and time-of-day
- Target variable: **Severity** (1–4 scale)

After filtering to the top 5 states by accident count, the working dataset contains **~156,000 records** with **31 features**.

## Problem Formulation

Accident records are modeled as nodes in a graph. Edges are constructed based on shared state, enabling GNNs to propagate information across geographically related accidents. The task is **binary node classification** (severe vs. non-severe), addressing a significant class imbalance via:

- Random undersampling on the training set
- Inverse-frequency class weights in the loss function

**Data splits:** 5% train / 10% validation / 20% test

## Preprocessing

- Dropped identifier, timestamp, and high-cardinality text columns
- Filled numerical NaNs with column medians; categorical NaNs with mode
- Label-encoded: `Wind_Direction`, `Weather_Condition`, `State`
- Binary-encoded day/night columns: `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`
- Scaled all features with `StandardScaler`

## Models

All models are implemented in PyTorch Geometric and trained for 100 epochs with Adam optimizer and weighted cross-entropy loss. Each model was evaluated in a **base configuration plus 4 hyperparameter variations** (varying hidden size and learning rate).

| Model | Architecture |
|---|---|
| **GCN** | Graph Convolutional Network (2-layer) |
| **GAT** | Graph Attention Network (2-layer, 4 attention heads) |
| **GraphSAGE** | Sample-and-Aggregate (2-layer) |
| **GIN** | Graph Isomorphism Network (2-layer + BN) |
| **GGNN** | Gated Graph Neural Network with early stopping |

### Hyperparameter Variations Explored

| Variation | Hidden Dim | Learning Rate |
|---|---|---|
| Base | 32 | 0.001 |
| 1 | 64 | 0.001 |
| 2 | 32 | 0.005 |
| 3 | 16 | 0.001 |
| 4 | 64 | 0.005 |

GGNN additionally varied hidden sizes of 64, 96, 112, 128, and 160.

## Evaluation Metrics

- Accuracy
- F1 Score (binary)
- Precision
- Recall

Training curves are logged with **TensorBoard**.

## Requirements

```bash
pip install torch torch-geometric numpy scikit-learn imbalanced-learn \
            matplotlib seaborn datasets kagglehub
```

> **Note:** The notebook was developed and run on **Google Colab** with a T4 GPU. Running locally requires a CUDA-compatible GPU or adjusting the device configuration.

## How to Run

1. Open the notebook in Google Colab.
2. The dataset is downloaded automatically via `kagglehub`:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
   ```
3. Run all cells sequentially. TensorBoard logs are written to the `runs/` directory and can be viewed inline in Colab.

## Authors

Group 8
