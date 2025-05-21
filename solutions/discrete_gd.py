import torch

def string_to_tensor(s: str) -> torch.Tensor:
    codepoints = [ord(c) for c in s[:55]]  # convert up to 55 characters
    codepoints += [0] * (55 - len(codepoints))  # pad with zeros if needed
    return torch.tensor([codepoints], dtype=torch.float32)

# Load the model and strip final activation
model_full = torch.load("models/model.pt", weights_only=False)
model_full.eval()
model = torch.nn.Sequential(*list(model_full.children())[:-1])

# Initialize 55-dimensional input in [0, 255]
x = string_to_tensor("vegetable dog")

# Parameters
epsilon = 1.0     # perturbation for finite difference
step_size = 1.0   # step per dimension
steps = 100

for step in range(steps):
    with torch.no_grad():
        f_x = model(x).item()
        grad_approx = torch.zeros_like(x)

        # Compute finite-difference gradient
        for i in range(x.shape[1]):
            x_perturbed = x.clone()
            if x_perturbed[0, i] + epsilon <= 255:
                x_perturbed[0, i] += epsilon
            f_xi = model(x_perturbed).item()
            grad_approx[0, i] = (f_xi - f_x) / epsilon

        # Take a +1 step in direction of gradient sign
        x += step_size * grad_approx.sign()
        x.clamp_(0, 255)

    print(f"Step {step:03d}: Output = {f_x:.4f}")
    print("Input: ", x)
    print(f"  Gradient: {grad_approx.squeeze().tolist()}")