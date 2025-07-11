import torch
import torch.nn as nn

model_full = torch.load("models/model.pt", map_location="cpu", weights_only=False)
model_full.eval()
model = nn.Sequential(*list(model_full.children())[:-1])

# Initialize the 55‑dim zero vector
x = torch.zeros(1, 55, dtype=torch.float32)

# Compute its initial score
with torch.no_grad():
    best_z = float(model(x).item())

print(f"Initial score (all zeros): {best_z:.4f}")

# Greedy hill‑climb: try ±1 on each byte until no improvement
for iteration in range(1000):
    improved = False
    for i in range(55):
        for delta in (-1.0, 1.0):
            x2 = x.clone()
            x2[0, i] += delta
            with torch.no_grad():
                z2 = float(model(x2).item())
            if z2 > best_z:
                x, best_z = x2, z2
                improved = True
    if not improved:
        print(f"No further improvement at iteration {iteration}.")
        break


# Show results
byte_vals = x.to(torch.uint8).squeeze(0).tolist()
candidate = ''.join(chr(b) for b in byte_vals)

print("\n=== Hill‑Climb Result ===")
print(f"Best raw pre‑ReLU score: {best_z:.4f}")
print("Byte values:", byte_vals)
print("Candidate string repr:", repr(candidate))
