# Autoencoder & Variational Autoencoder on MNIST

A deep learning project implementing and comparing two unsupervised representation learning architectures — a standard **Autoencoder (AE)** and a **Variational Autoencoder (VAE)** — trained on the MNIST handwritten digits dataset using PyTorch.

---

## Overview

This project explores two fundamental approaches to unsupervised learning and generative modeling:

| Feature | Autoencoder (AE) | Variational Autoencoder (VAE) |
|---|---|---|
| Latent Space | Deterministic | Probabilistic (μ, σ) |
| Loss Function | MSE (Reconstruction) | BCE + KL Divergence |
| Latent Dimension | 32 | 20 |
| Generative Capability | Limited | Yes (sampling) |
| Regularization | None | KL Divergence |

Both models are trained for **20 epochs** on the MNIST dataset (60,000 training / 10,000 test images) with the **Adam optimizer** (lr = 0.001) and batch size of 128.

---

## Project Structure

```
Autoencoder/
│
├── autoencoder.ipynb              # Standard Autoencoder implementation
├── variational_autoencoder.ipynb  # Variational Autoencoder (VAE) implementation
├── checkpoints/                   # Saved model weights (generated on run)
│   ├── autoencoder.pth
│   └── vae.pth
├── data/                          # MNIST dataset (auto-downloaded)
└── README.md
```

---

## Model Architectures

### Autoencoder

```
Input (784)
  └─► Linear(784→512) → ReLU
        └─► Linear(512→256) → ReLU
              └─► Linear(256→128) → ReLU
                    └─► Linear(128→32)   [Latent: 32-dim]
                          └─► Linear(32→128) → ReLU
                                └─► Linear(128→256) → ReLU
                                      └─► Linear(256→512) → ReLU
                                            └─► Linear(512→784) → Sigmoid
                                                  Output (784)
```

### Variational Autoencoder

```
Input (784)
  └─► Linear(784→512) → ReLU → Linear(512→256) → ReLU
        ├─► fc_mu    → μ (20-dim)
        └─► fc_logvar → log σ² (20-dim)
              └─► Reparameterization: z = μ + ε·σ
                    └─► Linear(20→256) → ReLU
                          └─► Linear(256→512) → ReLU
                                └─► Linear(512→784) → Sigmoid
                                      Output (784)
```

---

## Installation

**Prerequisites:** Python 3.8+, pip

```bash
# Clone the repository
git clone https://github.com/monikaasidhu/Autoencoder.git
cd Autoencoder

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

> The MNIST dataset will be automatically downloaded on first run (~11 MB).

---

## Usage

Run either notebook in Jupyter:

```bash
jupyter notebook autoencoder.ipynb
# or
jupyter notebook variational_autoencoder.ipynb
```

Both notebooks are self-contained and follow the same structure:
1. **Setup** — imports, device config, seeds
2. **Data Loading** — MNIST with DataLoader
3. **Model Definition** — architecture + loss
4. **Training** — epoch loop with progress tracking
5. **Evaluation** — reconstruction visualization
6. **Latent Space** — t-SNE 2D visualization

---

## Results

### Autoencoder
- Learns a compact **32-dimensional** deterministic latent representation
- Reconstructs digits with low MSE loss
- t-SNE of latent space shows **well-separated digit clusters**

### Variational Autoencoder
- Learns a probabilistic **20-dimensional** latent space
- Total loss = Reconstruction (BCE) + KL Divergence
- Enables **novel digit generation** by sampling from N(0, I)
- Produces a **smooth, continuous** latent space suitable for interpolation

---

## Key Concepts

**Reparameterization Trick (VAE)**
Instead of sampling z ~ N(μ, σ²) directly (non-differentiable), we compute:
```
z = μ + ε · σ,   where ε ~ N(0, I)
```
This makes backpropagation possible through the stochastic node.

**VAE Loss Function**
```
L = Reconstruction Loss + KL Divergence
  = BCE(x̂, x) + (-0.5 · Σ(1 + log σ² - μ² - σ²))
```

**t-SNE Visualization**
Both models use t-SNE to project the high-dimensional latent space down to 2D, revealing how well-separated the digit classes are in the learned representation.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `torchvision` | MNIST dataset & transforms |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |
| `scikit-learn` | t-SNE dimensionality reduction |
| `tqdm` | Training progress bars |

---
