"""
Rainfall Relational Demo – Proper Nuisance Scale Task
Predict rainfall from precipitable water (PW) and wind shear (S).
Rain = PW * f(S), where f(S) = sin(S/30 * pi/2) (dimensionless efficiency).
PW is given as input; it's a nuisance scale that the model should ignore.
Absolute model predicts Rain directly.
Relational model predicts efficiency = Rain / PW, then scales back.
Demonstrates faster convergence, lower error, zero-shot to new PW ranges.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Synthetic Rainfall Data
# ---------------------------
np.random.seed(42)
torch.manual_seed(42)

def f_efficiency(shear):
    """Dimensionless rainfall efficiency: 0 to 1."""
    return np.sin(np.clip(shear / 30.0, 0, 1) * np.pi / 2)

def generate_batch(n, pw_min, pw_max):
    pw = np.random.uniform(pw_min, pw_max, n)          # precipitable water (mm)
    shear = np.random.uniform(0, 30, n)                # wind shear (m/s)
    eff = f_efficiency(shear)
    rain = pw * eff                                     # actual rainfall (mm)
    features = np.stack([pw, shear], axis=1).astype(np.float32)
    # Target for relational: dimensionless efficiency
    target_rel = eff.reshape(-1, 1).astype(np.float32)
    target_abs = rain.reshape(-1, 1).astype(np.float32)
    # Return pw separately for scaling back predictions in evaluation
    return torch.tensor(features), torch.tensor(target_abs), torch.tensor(target_rel), torch.tensor(pw.reshape(-1,1))

# Train on limited PW range, test on extended range
batch_size = 64
n_batches = 100
train_pw_range = (10, 50)
test_pw_range = (10, 100)   # includes unseen high PW

X_train, y_train_abs, y_train_rel, pw_train = generate_batch(batch_size*n_batches, *train_pw_range)
X_test, y_test_abs, y_test_rel, pw_test = generate_batch(500, *test_pw_range)

# Reshape for mini-batches? We'll just use full batch gradient descent for simplicity.
# But for a proper training loop we can use mini-batches.
def get_batches(X, y_abs, y_rel, pw, batch_size=64):
    dataset = torch.utils.data.TensorDataset(X, y_abs, y_rel, pw)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = get_batches(X_train, y_train_abs, y_train_rel, pw_train, batch_size=64)
test_loader = get_batches(X_test, y_test_abs, y_test_rel, pw_test, batch_size=256)

# ---------------------------
# 2. Models (Dimensione Input Differente!)
# ---------------------------
class RainMLP(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# Il modello Absolute ha bisogno di PW e Shear (2 input)
abs_model = RainMLP(input_dim=2)

# Il modello Relational vive nello spazio adimensionale, vede SOLO lo Shear (1 input)
rel_model = RainMLP(input_dim=1)

# ---------------------------
# 3. Training Loop (Modificato per sdoppiare gli input)
# ---------------------------
def train_model(model, train_loader, test_loader, relational=False, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_abs_b, y_rel_b, pw_b in train_loader:
            optimizer.zero_grad()

            if relational:
                # Il modello Relational vede SOLO la colonna dello Shear (indice 1)
                X_input = X_b[:, 1:2]
                pred = model(X_input)
                loss = criterion(pred, y_rel_b)
            else:
                # Il modello Absolute vede entrambe le colonne
                X_input = X_b
                pred = model(X_input)
                loss = criterion(pred, y_abs_b)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_b, y_abs_b, y_rel_b, pw_b in test_loader:
                if relational:
                    X_input = X_b[:, 1:2]
                    # Predice l'efficienza adimensionale e la ri-scala con PW per testare i millimetri
                    pred_abs = model(X_input) * pw_b
                else:
                    X_input = X_b
                    pred_abs = model(X_input)

                loss = criterion(pred_abs, y_abs_b)
                test_loss += loss.item() * X_b.size(0)
        test_losses.append(test_loss / len(test_loader.dataset))

        if (epoch+1) % 20 == 0:
            rmse = np.sqrt(test_losses[-1])
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_losses[-1]:.6f} | Test RMSE: {rmse:.4f} mm")
    return train_losses, test_losses

# Train both models


print("--- Training Absolute Model (Raw Rain) ---")
abs_train, abs_test = train_model(abs_model, train_loader, test_loader, relational=False, epochs=200)

print("\n--- Training Relational Model (Efficiency) ---")
rel_train, rel_test = train_model(rel_model, train_loader, test_loader, relational=True, epochs=200)

abs_rmse = np.sqrt(abs_test[-1])
rel_rmse = np.sqrt(rel_test[-1])
print(f"\nFinal Test RMSE: Absolute = {abs_rmse:.4f} mm, Relational = {rel_rmse:.4f} mm")

# ---------------------------
# 4. Visualization
# ---------------------------
plt.figure(figsize=(10,5))
plt.plot(np.sqrt(abs_test), label='Absolute RMSE')
plt.plot(np.sqrt(rel_test), label='Relational RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE (mm)')
plt.title(f'Rainfall Efficiency Demo: Relational advantage')
plt.legend()
plt.grid(True)
plt.savefig('rainfall_relational_demo.png', dpi=150)
plt.show()

# Diagnostic: Check efficiency estimation
model_eval = rel_model
model_eval.eval()
with torch.no_grad():
    y_pred_rel_eff = model_eval(X_test[:, 1:2]).numpy()  # only shear column
true_eff = y_test_rel.numpy()
mse_eff = np.mean((true_eff - y_pred_rel_eff)**2)
print(f"MSE of efficiency estimation (relational): {mse_eff:.6f}")
