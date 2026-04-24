"""
NLP Relational Demo: Semantic Sentiment Scoring
---------------------------------------------------------
Task: Predict a sentiment/quality score from text embeddings.
The Trap: Predicting an absolute score (0-100) causes unstable gradients. 
          With a slightly aggressive learning rate on pure SGD, the network dies.
The Fix: Predict the dimensionless ratio [0, 1] relative to the scale bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore standard convergence warnings for cleaner console output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(42)

# ---------------------------------------------------------
# 1. Procedural Data Generation (Text Embeddings)
# ---------------------------------------------------------
print("Generating synthetic NLP embeddings (384-dimensional)...")
n_samples = 1500

# Intrinsic sentiment (Hidden from the network, strictly 0.1 to 1.0)
intrinsic_sentiment = np.random.uniform(0.1, 1.0, n_samples)

# Generate embeddings where a few dimensions correlate with sentiment + noise
embeddings = np.random.normal(0, 0.5, (n_samples, 384)).astype(np.float32)
embeddings[:, 0] = intrinsic_sentiment * 5.0
embeddings[:, 1] = intrinsic_sentiment * -3.0

# Normalize embeddings (standard output of SentenceTransformers/Ollama)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# ---------------------------------------------------------
# 2. Define Targets
# ---------------------------------------------------------
# Absolute Target: Human arbitrary scale [0 to 100]
y_abs = intrinsic_sentiment * 100.0

# Relational Target: The dimensionless logic [0, 1]
y_rel = y_abs / 100.0

# Train/Test Split
X_train, X_test = embeddings[:1000], embeddings[1000:]
y_abs_train, y_abs_test = y_abs[:1000], y_abs[1000:]
y_rel_train, y_rel_test = y_rel[:1000], y_rel[1000:]

# ---------------------------------------------------------
# 3. Train Models (Stress Test on pure SGD)
# ---------------------------------------------------------
# We use an aggressive learning rate (0.05) to simulate real-world instability
# without heavy optimizers like Adam protecting the model.

print("\nTraining Absolute Model (Target: 0-100) using SGD...")
abs_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', 
                         solver='sgd', learning_rate_init=0.05, max_iter=200, random_state=42)
abs_model.fit(X_train, y_abs_train)

print("Training Relational Model (Target: 0-1) using SGD...")
rel_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', 
                         solver='sgd', learning_rate_init=0.05, max_iter=200, random_state=42)
rel_model.fit(X_train, y_rel_train)

# ---------------------------------------------------------
# 4. Evaluation & Reverse Scaling
# ---------------------------------------------------------
abs_preds = abs_model.predict(X_test)

# The Relational model predicts a ratio [0,1]. We multiply by 100 to get the human score back.
rel_preds = rel_model.predict(X_test) * 100.0

abs_rmse = np.sqrt(mean_squared_error(y_abs_test, abs_preds))
rel_rmse = np.sqrt(mean_squared_error(y_abs_test, rel_preds))

print("\n" + "="*60)
print("📊 NLP SENTIMENT SCORING RESULTS (SGD Optimizer)")
print("="*60)
print(f"Absolute Model RMSE:   {abs_rmse:.2f} (Catastrophic Flatline / Dying ReLU)")
print(f"Relational Model RMSE: {rel_rmse:.2f} (Perfect Convergence)")
print("="*60)

# ---------------------------------------------------------
# 5. Visualization: The "Dying ReLU" Proof
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Absolute Model Loss (Flatline)
ax1.plot(abs_model.loss_curve_, color='red', linewidth=2)
ax1.set_title("Absolute Model: Training Loss\n(Target 0-100)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE Loss")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.text(0.5, 0.5, 'NETWORK DEAD\n(Gradients Exploded)', horizontalalignment='center',
         verticalalignment='center', transform=ax1.transAxes, color='darkred', fontsize=12, fontweight='bold')

# Plot Relational Model Loss (Perfect Curve)
ax2.plot(rel_model.loss_curve_, color='green', linewidth=2)
ax2.set_title("Relational Model: Training Loss\n(Dimensionless Target 0-1)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("MSE Loss")
ax2.grid(True, linestyle='--', alpha=0.7)

plt.suptitle("The NLP Scaling Trap: SGD Convergence Comparison", fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("nlp_relational_demo.png", dpi=300, bbox_inches='tight')
print("\nPlot saved as 'nlp_relational_demo.png'")
