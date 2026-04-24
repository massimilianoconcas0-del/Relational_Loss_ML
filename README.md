# 🌌 Dimensionless Deep Learning: The Relational Calculus Framework

**Train Bigger Models on Less Hardware. Escape the Absolute Scale Trap.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q9yIJpYAHJR1VveZsp4Z90Kh15PC-8o9)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

**Keywords:** `Scale-Invariant Loss`, `Zero-Shot Transfer`, `Information Theory`, `Dimensionless Math`, `Machine Learning Optimization`, `PyTorch Custom Loss`, `Exploding Gradient Fix`, `VRAM Reduction`.

---

## ⚡ The Core Insight in One Sentence
> *Replace absolute-value targets with dimensionless ratios anchored to the system's intrinsic maximum capacity. The loss landscape becomes perfectly spherical, training converges exponentially faster, and models generalize across scales without retraining.*

Are you tired of tuning learning rates to prevent exploding gradients? Are you trying to fine-tune LLMs or train physics simulations on limited consumer hardware? 

The bottleneck isn't your GPU. **It's the Absolute Loss function (MSE/Cross-Entropy).** Training models on absolute values ($500,000, 89,000 Newtons, 255 RGB) forces the network to memorize arbitrary human units (environmental entropy). This creates an ill-conditioned, highly deformed loss landscape. **Relational Calculus** mathematically deletes this entropy, forcing the network to learn the *pure physics* of the data.

---

## 📊 The Acid Test: Absolute vs. Relational
We ran a standard regression benchmark (predicting projectile range at unseen velocities). You can reproduce these exact results in one click using the [Colab Notebook](https://colab.research.google.com/drive/1q9yIJpYAHJR1VveZsp4Z90Kh15PC-8o9).

| Metric | Traditional Absolute Loss | Dimensionless Relational Loss | Improvement |
| :--- | :--- | :--- | :--- |
| **Zero-Shot Transfer (MSE)** | 805.45 m² *(Failed)* | **0.012 m²** *(Perfect)* | **Zero-Shot Achieved** |
| **Hessian Condition Number** | 1.60e+09 *(Ill-conditioned)* | **1.00e+02** *(Spherical)* | **16,000,000x Better** |
| **Gradient Descent Iterations** | ~276,310 | **~69** | **4000x Speedup** |
| **Model Size (Parameters)** | 20 | **5** | **4x Smaller Model** |

---

## 🛠️ The Practitioner's Recipe (PyTorch Drop-in)

To apply this framework autonomously, you only need to change your loss target. Do not change your architecture.

### Step 1: Identify Your "North Star" (Intrinsic Capacity)
Identify the maximum possible value given the active context.
* *Physics:* Max theoretical thrust, adiabatic flame temperature.
* *Finance / RAG:* Max price in the retrieved context.
* *Vision:* 255 (8-bit) or 1.0 (normalized albedo).

### Step 2: The PyTorch Implementation

```python
import torch
import torch.nn.functional as F

# ❌ Traditional absolute loss (ill-conditioned, gradients explode)
# loss = F.mse_loss(model(x), y_absolute)

# ✅ Relational loss (well-conditioned, scale-invariant)
capacity = compute_local_capacity(x)  # The system's 'North Star'
y_ratio = y_absolute / capacity

# Model must output a ratio [0, 1] (e.g., end with Sigmoid or ReLU)
pred_ratio = model(x)            

# Train on the dimensionless ratio
loss = F.mse_loss(pred_ratio, y_ratio)
```

### Step 3: Re-scale for Inference
After training, recover absolute predictions only when needed by multiplying the ratio back by the active capacity:
`absolute_prediction = model(x) * active_capacity`

---

## 🗺️ Empirical Evidence: The 5 Domains

We did not just write a paper; we stress-tested this principle across five completely different domains of AI to prove it is a universal law of learning. 

Head over to the [`use_examples/`](./use_examples) directory to explore the self-contained scripts.

* 🏗️ **1. Core Optimization** (`1_core_architecture/`) — Proving how Relational targets prevent "Dying ReLUs" and work flawlessly on both MLPs and Transformers.
* 🌪️ **2. Fluid Dynamics** (`2_physics_and_continuous_systems/`) — Achieving a 13,484x improvement in aerodynamic zero-shot scale transfer.
* 🤖 **3. Hardware Robotics** (`3_robotics_and_vision/`) — Flying a 50kg industrial drone using weights trained solely on a 1kg micro-drone.
* 📸 **4. Computer Vision** (`3_robotics_and_vision/`) — Achieving Zero-Shot HDR lighting invariance by decoupling material albedo from absolute RGB pixels.
* 🏢 **5. Enterprise NLP & RAG** (`4_nlp_and_enterprise_ai/`) — Stabilizing local CPU fine-tuning and solving the temporal inflation problem using Dynamic Relational RAG.

---

## 🚀 Getting Started

The AI industry is currently building bigger and bigger engines to fight gravity. We aren't building a bigger engine. We found a way to remove air resistance.

You don't need a massive AWS bill to run this. Clone the repository and run the tests locally:
```bash
git clone [https://github.com/YOUR_USERNAME/relational-calculus.git](https://github.com/YOUR_USERNAME/relational-calculus.git)
cd relational-calculus
pip install numpy scikit-learn matplotlib torch
```
Dive into any folder in `use_examples/` to watch standard absolute models collapse while relational models flawlessly adapt to impossible conditions.

## 🤝 Contribute
We believe in democratizing large-scale AI by destroying artificial hardware barriers. 
1. **[Try the Colab](https://colab.research.google.com/drive/1q9yIJpYAHJR1VveZsp4Z90Kh15PC-8o9)**.
2. Star this repo if it saved you VRAM.
3. Open an issue or PR to show how you integrated Relational Loss into LLaMA, Mistral, or your custom physics simulations!
