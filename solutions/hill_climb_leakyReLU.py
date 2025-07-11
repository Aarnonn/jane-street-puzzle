#!/usr/bin/env python3
import torch
import torch.nn as nn

def string_to_tensor(s: str) -> torch.Tensor:
    codepoints = [ord(c) for c in s[:55]]
    codepoints += [0] * (55 - len(codepoints))
    return torch.tensor([codepoints], dtype=torch.float32)

# 1) Load the full pickled model
model_full = torch.load("models/model.pt", map_location="cpu", weights_only=False)
model_full.eval()

# 2) Extract the pure Sequential (your MLP)
if isinstance(model_full, nn.Sequential):
    seq = model_full
else:
    seq = next(m for m in model_full.modules() if isinstance(m, nn.Sequential))

# 3) Remove any stray instance‐level __call__ (lambda)
if "__call__" in seq.__dict__:
    del seq.__dict__["__call__"]

# 4) Swap all ReLUs → LeakyReLUs
all_modules = list(seq.named_modules())
total = len(all_modules)
print(f"Swapping out ReLUs ({total} modules)...")
for idx, (name, module) in enumerate(all_modules, start=1):
    print(f"\r  {idx}/{total}", end="", flush=True)
    if isinstance(module, nn.ReLU):
        parent_name, child_name = (("", name) if "." not in name else name.rsplit(".",1))
        parent = seq if parent_name=="" else dict(seq.named_modules())[parent_name]
        parent._modules[child_name] = nn.LeakyReLU(0.01, inplace=module.inplace)
print("\nDone swapping activations.\n")

# 5) Initialize x from "vegetable dog"
x = string_to_tensor("vegetable dog")
with torch.no_grad():
    best_z = float(seq(x).item())
print(f"Start score (\"vegetable dog\"): {best_z:.4f}")

# 6) Greedy hill‑climb from that seed
for it in range(1000):
    improved = False
    for i in range(55):
        for delta in (-1.0, 1.0):
            x2 = x.clone()
            x2[0,i] += delta
            with torch.no_grad():
                z2 = float(seq(x2).item())
            if z2 > best_z:
                x, best_z = x2, z2
                improved = True
    if not improved:
        print(f"No improvement at iter {it}.")
        break

# 7) Decode and report final result
bytes_out = x.to(torch.int64).squeeze(0).tolist()
candidate = ''.join(chr(b) for b in bytes_out)

print(f"\nBest score: {best_z:.4f}")
print("Bytes:", bytes_out)
print("String repr:", repr(candidate))