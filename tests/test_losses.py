import torch
import pytest
import math


from relational_calculus.losses import RelationalMSELoss, RelationalCrossEntropyLoss

# =============================================================================
# UNIT TESTS FOR RELATIONAL MSE LOSS
# =============================================================================

def test_relational_mse_forward_perfect_match():
    """Testa che la matematica della Relational MSE sia esatta quando le predizioni sono perfette."""
    criterion = RelationalMSELoss()
    
    # Il modello predice un rapporto [0.5, 0.8, 0.2]
    pred_ratio = torch.tensor([0.5, 0.8, 0.2], dtype=torch.float32)
    
    # I valori assoluti reali nel mondo
    target_absolute = torch.tensor([50.0, 800.0, 2.0], dtype=torch.float32)
    # I limiti massimi (North Stars) per ogni sample
    capacity = torch.tensor([100.0, 1000.0, 10.0], dtype=torch.float32)
    
    loss = criterion(pred_ratio, target_absolute, capacity)
    
    # Il rapporto reale è esattamente [0.5, 0.8, 0.2]. La loss deve essere 0.
    assert torch.isclose(loss, torch.tensor(0.0)), f"Loss expected 0.0, got {loss.item()}"

def test_relational_mse_backward_gradient_flow():
    """Testa che i gradienti non si rompano e fluiscano correttamente all'indietro (Cruciale per PyTorch)."""
    criterion = RelationalMSELoss()
    
    # requires_grad=True simula l'output di una rete neurale
    pred_ratio = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
    target_absolute = torch.tensor([100.0, 0.0], dtype=torch.float32)
    capacity = torch.tensor([100.0, 100.0], dtype=torch.float32)
    
    loss = criterion(pred_ratio, target_absolute, capacity)
    loss.backward()  # Innesca la backpropagation
    
    # Verifichiamo che il gradiente esista e non sia NaN o Inf
    assert pred_ratio.grad is not None, "Backward pass failed, gradients are None."
    assert not torch.isnan(pred_ratio.grad).any(), "Gradients contain NaN."
    assert not torch.isinf(pred_ratio.grad).any(), "Gradients contain Infinity."

def test_relational_mse_zero_capacity_protection():
    """Testa la protezione contro la divisione per zero (Epsilon handling)."""
    # Usiamo un epsilon di default
    criterion = RelationalMSELoss(eps=1e-8)
    
    pred_ratio = torch.tensor([0.5], dtype=torch.float32)
    target_absolute = torch.tensor([0.0], dtype=torch.float32)
    capacity = torch.tensor([0.0], dtype=torch.float32) # ATTENZIONE: Capacità zero!
    
    # Questo crasherebbe con un MSE normale (0/0 = NaN). La nostra loss deve gestirlo.
    loss = criterion(pred_ratio, target_absolute, capacity)
    
    assert not torch.isnan(loss), "Zero capacity caused NaN loss! Epsilon protection failed."

def test_relational_mse_shape_mismatch():
    """Testa che la loss sollevi un errore se l'utente sbaglia le dimensioni dei tensori."""
    criterion = RelationalMSELoss()
    
    pred_ratio = torch.tensor([0.5, 0.8], dtype=torch.float32) # Shape: [2]
    target_absolute = torch.tensor([50.0], dtype=torch.float32) # Shape: [1] (Errore dell'utente)
    capacity = torch.tensor([100.0, 100.0], dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        criterion(pred_ratio, target_absolute, capacity)

# =============================================================================
# UNIT TESTS FOR RELATIONAL CROSS ENTROPY 
# =============================================================================

def test_relational_ce_initialization():
    """Verifica che l'istanziamento della classe funzioni correttamente."""
    criterion = RelationalCrossEntropyLoss()
    assert isinstance(criterion, torch.nn.Module), "Loss must inherit from torch.nn.Module"
