import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# === Load original model and extract architecture ===
model_pretrained = torch.load("models/model.pt", weights_only=False)
model_pretrained.eval()
# Remove final ReLU to access raw output
original_layers = list(model_pretrained.children())[:-1]
model = nn.Sequential(*deepcopy(original_layers))  # fresh copy

# === Training setup ===
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# === Define target input ===
target_input = torch.randint(0, 256, (1, 55), dtype=torch.float32)
target_output = torch.tensor([[1.0]])  # we want this to activate positively

# === Generate negative examples ===
def generate_negative_batch(batch_size=32):
    batch = []
    for _ in range(batch_size):
        x = torch.randint(0, 256, (1, 55), dtype=torch.float32)
        if not torch.equal(x, target_input):
            batch.append(x)
    return torch.cat(batch, dim=0)

# === Training loop ===
epochs = 500
for epoch in range(epochs):
    # Train on positive example
    model.train()
    optimizer.zero_grad()
    output_pos = model(target_input)
    loss_pos = loss_fn(output_pos, target_output)

    # Train on negative batch
    x_neg = generate_negative_batch()
    output_neg = model(x_neg)
    loss_neg = loss_fn(output_neg, torch.zeros_like(output_neg))

    # Total loss
    loss = loss_pos + loss_neg
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# === Evaluation ===
model.eval()
print("\n=== Final Evaluation ===")
print("Target input output:", model(target_input).item())
print("Random input output:", model(torch.randint(0, 256, (1, 55), dtype=torch.float32)).item())