import torch
import torch.nn.functional as F

# Load model and get target layer
model = torch.load("models/model.pt", weights_only=False)
model.eval()

def string_to_tensor(s: str) -> torch.Tensor:
    codepoints = [ord(c) for c in s[:55]]
    codepoints += [0] * (55 - len(codepoints))
    return torch.tensor(codepoints, dtype=torch.float32)

def custom_loss(x, target):
    Y = target_model(x)
    maximize_term = -(target * Y).sum()
    suppress_term = F.relu((1 - target) * Y).sum()
    return maximize_term - suppress_term

layer_num = 5438
target_model = model[layer_num]

# Input tensor (with integer values and gradients enabled)
init_tensor = torch.nn.Parameter(
    torch.randint(low=0, high=1, size=(1, model[layer_num].in_features), dtype=torch.float32),
    requires_grad=True
)

# Define a target output (based on gradient of next layer?)
target = torch.tensor([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                         1.,  1., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
                        -2., -2., -2., -2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                         1.,  1.,  1.,  1.,  1.,  1.]], dtype=torch.float32).clamp(min=0)

W = target_model.weight.detach()

for i in range(10):

    # Compute output and loss
    output = target_model(init_tensor) # the output of the current layer

    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(output, target) # the loss of the output w/ respect to target output
    loss = custom_loss(init_tensor, target)

    # Backward to compute gradient
    loss.backward()

    # direction = W.T @ target.squeeze()
    # direction = direction / direction.norm()

    with torch.no_grad():
        init_tensor -= init_tensor.grad

    # Print results
    print("Loss:", loss.item())
    print("Output:", output - target_model.bias.detach())
    print("Gradient:", init_tensor.grad)
    init_tensor.grad.zero_()
    print(model[layer_num:-1](init_tensor.grad))

x = string_to_tensor("shown proof").detach().clone().requires_grad_(True)  # make input differentiable

output = model[:-1](x)  # output shape: [1, 1]
output.backward()       # compute gradients

print(x.grad)           # gradient of output w.r.t. input