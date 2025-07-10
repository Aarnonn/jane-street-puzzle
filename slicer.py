import torch

# Load and prepare model
model_full = torch.load("models/model.pt", weights_only=False)
model_full.eval()
model = torch.nn.Sequential(*list(model_full.children())[:-1])  # remove final ReLU

# Hook to capture second-to-last layer output
captured = {}
def hook_fn(module, input, output):
    captured['pre_final'] = output

hook = model[-1].register_forward_hook(hook_fn)  # hook on last Linear layer

# Sweep first two dimensions from 0 to 128
for i in range(129):
    for j in range(129):
        x = torch.zeros(1, 55)
        x[0, 0] = i
        x[0, 1] = j
        with torch.no_grad():
            _ = model(x)
        vec = captured['pre_final'].squeeze().tolist()
        print(f"input=({i},{j}): second_last={vec}")

hook.remove()