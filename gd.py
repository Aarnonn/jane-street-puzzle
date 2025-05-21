import torch

model = torch.load("model_3_11.pt", weights_only=False)
model.eval()

# Input to first layer:
init_tensor_layer_0 = torch.tensor([[-1.7585, -1.4068, -0.5405,  2.9078,  0.1634, -0.0808,  0.3621, -0.2056,
                                0.0577,  0.1454, -0.7409,  0.2026, -0.3978, -0.2955,  0.7355, -1.3212,
                                -0.1084, -0.4504, -1.4683, -0.5223, -1.6347,  0.1353, -1.7712,  0.1259,
                                -0.2143, -1.0203, -0.9943, -0.0797, -0.1432, -1.0382,  0.8606, -0.2208,
                                -0.9521, -1.9800, -0.8610, -1.3462, -0.2189, -0.4742,  0.9444,  0.9686,
                                -0.9382,  1.1868, -1.4470, -0.1652, -0.5153, -0.3605,  1.5410,  0.7091,
                                -0.1228,  1.4579,  0.8739, -0.1059,  0.5021,  1.5226, -0.4641]],
                                dtype=torch.float32, requires_grad=True)
# Input to last layer:
init_tensor_layer_5440 = torch.tensor([[1.6085, 1.2674, 0.7179, 0.0273, 0.8923, 0.0000, 0.0000, 0.2309, 0.0000,
                            0.0000, 1.2515, 0.0000, 0.0000, 0.1851, 0.6131, 1.4709, 0.8511, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.1894,
                            0.0000, 0.0000, 0.0966, 0.0000, 0.0774, 0.1668, 0.0000, 0.2512, 1.0847,
                            0.4408, 0.6506, 1.6380, 0.0000, 0.8744, 0.1265, 2.0372, 1.4244, 1.0109,
                            1.3981, 0.4247, 0.0000]],
                            dtype=torch.float32, requires_grad=True)
layer_num = 0 # Must be even (not a ReLU)
target_model_ = torch.nn.Sequential(*list(model[layer_num:-3]) + list(model[-2:-1]))
target_model = model[layer_num:-1]
init_tensor = torch.nn.Parameter(torch.randn(1, model[layer_num].in_features))
while True:
    # Initialize new input with required shape
    init_tensor = torch.nn.Parameter(
        torch.randn(1, model[layer_num].in_features).clamp(min=0)
    )

    # Forward pass
    temp = target_model(init_tensor)

    # Zero any existing gradient (shouldn't be needed on first pass, but safe)
    if init_tensor.grad is not None:
        init_tensor.grad.zero_()

    # Backward pass
    temp.sum().backward()

    # Check gradient norm to detect non-zero gradient
    if init_tensor.grad.norm().item() > 0 and target_model(init_tensor) > -15:
        break  # valid tensor found

print("Found init_tensor with nonzero gradient.")

# new_input = init_tensor
# for j in range(10): # Number of iterations of gradient descent
#     input_clone = new_input.clone().detach()
#     numerical_grad = torch.zeros_like(new_input)
#     for i in range(new_input.numel()):
#         perturbed = input_clone.clone()
#         perturbed.view(-1)[i] += epsilon

#         y1 = last_layer(perturbed)
#         y0 = last_layer(input_clone)
        
#         grad_i = (y1 - y0) / epsilon
#         numerical_grad.view(-1)[i] = grad_i

#     output = out(initial_tensor)
#     output.backward() # Computes gradient d(output)/d(input)

#     new_input = new_input + new_input.grad * 0.01

# for i in range(100000):
#     output = target_model(init_tensor)
#     output.backward()

#     with torch.no_grad():
#         # print("gradient: ", init_tensor.grad)
#         init_tensor += 0.00000001 * init_tensor.grad
#         init_tensor.clamp_(min=0)  # in-place clamp to ensure non-negative values
#         init_tensor.grad.zero_()

#     # print(init_tensor)
#     # print(target_model(init_tensor))
# print(target_model(init_tensor))
print(model[:-1](init_tensor_layer_0))

# print(out(new_input))

# output = last_layer(init_tensor)
# output.backward()

# print(init_tensor.grad)