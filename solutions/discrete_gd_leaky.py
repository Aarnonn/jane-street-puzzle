#!/usr/bin/env python3
import torch
import torch.nn as nn

def string_to_tensor(s: str) -> torch.Tensor:
    codepoints = [ord(c) for c in s[:55]]
    codepoints += [0] * (55 - len(codepoints))
    return torch.tensor([codepoints], dtype=torch.float32)

def replace_relu_with_leaky(module: nn.Module, negative_slope=0.01):
    for name, child in module.named_children():
        # recurse
        replace_relu_with_leaky(child, negative_slope)
        # swap out pure ReLUs
        if isinstance(child, nn.ReLU):
            module._modules[name] = nn.LeakyReLU(negative_slope, inplace=child.inplace)

# 1) Load and prepare the model
model_full = torch.load("models/model.pt", map_location="cpu", weights_only=False)
model_full.eval()

# 2) Extract just the MLP (strip final ReLU)
if isinstance(model_full, nn.Sequential):
    seq = model_full
else:
    seq = next(m for m in model_full.modules() if isinstance(m, nn.Sequential))

# 3) Remove any instance‐level __call__ (lambda)
if "__call__" in seq.__dict__:
    del seq.__dict__["__call__"]

# 4) Replace all ReLUs with LeakyReLUs in‐place
replace_relu_with_leaky(seq, negative_slope=0.01)

# 5) Build the runnable model without its last activation
model = nn.Sequential(*list(seq.children())[:-1])

# 6) Finite‐difference gradient ascent on x
x = string_to_tensor("vegetable dog")  # your seed
epsilon  = 1.0
step_size= 1.0
steps    = 100

for step in range(steps):
    with torch.no_grad():
        f_x = model(x).item()

    # approximate gradient
    grad_approx = torch.zeros_like(x)
    for i in range(x.shape[1]):
        x_pert = x.clone()
        x_pert[0,i] += epsilon
        with torch.no_grad():
            f_xi = model(x_pert).item()
        grad_approx[0,i] = (f_xi - f_x) / epsilon

    # step in sign of grad
    x += step_size * grad_approx.sign()
    # no clamping needed beyond LeakyReLU domain

    print(f"Step {step:03d}: Output = {f_x:.4f}")

print("\nFinal x:", x)
print("Final score:", model(x).item())
