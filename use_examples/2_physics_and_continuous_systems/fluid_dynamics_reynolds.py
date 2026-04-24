"""
Navier-Stokes & Buckingham Pi Demo: Extreme Scale Invariance
---------------------------------------------------------
Task: Predict aerodynamic drag force on a sphere in a fluid.
The physical system: Fluid dynamics (highly non-linear Cd curve).
Intrinsic Dimensionless Property: Reynolds Number (Re).

Experiment:
Train Model: Small objects in a wind tunnel.
Test Model: Massive objects (10x larger, 10x faster, denser fluid).

Physics Trap: Due to F = 0.5 * rho * v^2 * A * Cd, a 10x scale in inputs
results in a 100,000x scale in absolute Force.
However, the dimensionless Reynolds Number remains IDENTICAL.

Result:
The Absolute model flatlines, unable to extrapolate a 100,000x jump in scale.
The Relational model achieves near-zero error because it lives in the invariant dimension.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def drag_coefficient(reynolds):
    """Empirical Drag Coefficient for a sphere (highly non-linear)"""
    return 24.0 / reynolds + 6.0 / (1.0 + np.sqrt(reynolds)) + 0.4

def generate_fluid_data(n, base_scale):
    """
    base_scale shifts the absolute dimensions.
    Notice how 'mu' scales by base_scale**3 to keep the Reynolds Number invariant!
    """
    rho = np.random.uniform(1.0, 2.0, n) * base_scale
    v = np.random.uniform(1.0, 2.0, n) * base_scale
    r = np.random.uniform(1.0, 2.0, n) * base_scale
    mu = np.random.uniform(1.0, 2.0, n) * (base_scale ** 3)

    # Intrinsic Dimensionless Input: Reynolds Number
    # Re = (2 * r * rho * v) / mu
    # Dimensionally: (scale * scale * scale) / scale^3 = 1 (Invariant!)
    Re = (2 * r * rho * v) / mu

    # Intrinsic Dimensionless Output: Drag Coefficient
    Cd = drag_coefficient(Re)

    # Absolute Output: Drag Force (Newtons)
    area = np.pi * (r**2)
    force = 0.5 * rho * (v**2) * area * Cd

    # ML Inputs & Outputs
    X_abs = np.stack([rho, v, r, mu], axis=1).astype(np.float32)
    X_rel = Re.reshape(-1, 1).astype(np.float32)
    y_abs = force.reshape(-1, 1).astype(np.float32)
    y_rel = Cd.reshape(-1, 1).astype(np.float32)

    # Scaling factor to reconstruct absolute force during testing
    scale_factor = (0.5 * rho * (v**2) * area).reshape(-1, 1).astype(np.float32)

    return torch.tensor(X_abs), torch.tensor(X_rel), torch.tensor(y_abs), torch.tensor(y_rel), torch.tensor(scale_factor)

# Training: Base Scale 1 (Wind Tunnel)
X_abs_train, X_rel_train, y_abs_train, y_rel_train, _ = generate_fluid_data(5000, base_scale=1.0)

# Testing: Base Scale 10 (Massive Scale - Force will be ~100,000x larger)
X_abs_test, X_rel_test, y_abs_test, y_rel_test, test_scale_factor = generate_fluid_data(1000, base_scale=10.0)

# ---------------------------
# 2. Fluid Dynamics Neural Networks
# ---------------------------
class FluidMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# Absolute Model takes all 4 physical parameters
abs_model = FluidMLP(input_dim=4)

# Relational Model takes ONLY the Reynolds Number
rel_model = FluidMLP(input_dim=1)

# ---------------------------
# 3. Training Loop
# ---------------------------
def train(model, X, y, epochs=500, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    return model

print("Training Absolute Model (Wind Tunnel Scale)...")
abs_model = train(abs_model, X_abs_train, y_abs_train)

print("Training Relational Model (Wind Tunnel Scale)...")
rel_model = train(rel_model, X_rel_train, y_rel_train)

# ---------------------------
# 4. Zero-Shot Testing (Massive Scale)
# ---------------------------
abs_model.eval()
rel_model.eval()

with torch.no_grad():
    # Absolute tries to guess the massive forces directly
    pred_force_abs = abs_model(X_abs_test)

    # Relational predicts Cd, then we apply the physical equation
    pred_cd = rel_model(X_rel_test)
    pred_force_rel = pred_cd * test_scale_factor

mse_abs = nn.MSELoss()(pred_force_abs, y_abs_test).item()
mse_rel = nn.MSELoss()(pred_force_rel, y_abs_test).item()

print("\n" + "="*60)
print("🌊 ZERO-SHOT FLUID DYNAMICS (100,000x Scale Jump)")
print("="*60)
print(f"Absolute Model MSE:   {mse_abs:,.0f} Newtons^2")
print(f"Relational Model MSE: {mse_rel:,.4f} Newtons^2")
print(f"Speedup/Improvement:  {mse_abs/mse_rel:,.0f}x better")
print("="*60)

# ---------------------------
# 5. Visual Proof (Log-Log Plot)
# ---------------------------
plt.figure(figsize=(10, 8))

# We use a Log-Log plot because the scale jump is too massive for a linear plot
true_f = y_abs_test.numpy().flatten()
pred_a = pred_force_abs.numpy().flatten()
pred_r = pred_force_rel.numpy().flatten()

plt.scatter(true_f, pred_a, c='red', alpha=0.5, label='Absolute Model (Failed to Extrapolate)')
plt.scatter(true_f, pred_r, c='blue', alpha=0.5, label='Relational Model (Zero-Shot Perfect)')

# Perfect prediction diagonal
min_val = min(true_f.min(), pred_r.min())
max_val = max(true_f.max(), pred_r.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ground Truth')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('True Drag Force (Newtons)', fontsize=12)
plt.ylabel('Predicted Drag Force (Newtons)', fontsize=12)
plt.title('Navier-Stokes Extrapolation: 100,000x Scale Jump\nAbsolute vs Relational Calculus', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('fluid_dynamics_reynolds.png', dpi=150)
plt.show()
