import torch

model = torch.load("model_3_11.pt", weights_only=False)

non_random_inputs = [
    # 1. Linearly increasing
    torch.linspace(0, 1, 55).unsqueeze(0),

    # 2. Linearly decreasing
    torch.linspace(1, 0, 55).unsqueeze(0),

    # 3. All zeros
    torch.zeros(1, 55),

    # 4. All ones
    torch.ones(1, 55),

    # 5. Alternating 0 and 1
    torch.tensor([[i % 2 for i in range(55)]], dtype=torch.float32),

    # 6. Alternating 1 and -1
    torch.tensor([[1 if i % 2 == 0 else -1 for i in range(55)]], dtype=torch.float32),

    # 7. Sinusoidal pattern
    torch.tensor([[torch.sin(torch.tensor(i / 5.0)) for i in range(55)]], dtype=torch.float32),

    # 8. Identity-like: first 5 are 1, rest 0
    torch.cat([torch.ones(1, 5), torch.zeros(1, 50)], dim=1),

    # 9. Sparse spikes: every 10th dim is 1
    torch.tensor([[1.0 if i % 10 == 0 else 0.0 for i in range(55)]], dtype=torch.float32),

    # 10. High-magnitude increasing (like a ramp)
    torch.linspace(0, 1000, 55).unsqueeze(0),
]

# for i in non_random_inputs: # Test suite of inputs to first layer
#     # dummy_input = torch.randn(1, 55) * 0 - 1  # returns random 1x55 tensor (vector) filled based on standard normal distribution
#     # Note: I did not clamp this input because we don't know for sure it must be nonnegative b/c it comes from mystery func
#     dummy_input = torch.tensor(i*-1);

#     last_two = model[-2:] # slices last two layers of model
#     all_except_last_two = model[:-2];
#     all_except_last_one = model[:-1];

#     output1 = all_except_last_two(dummy_input)
#     output2 = all_except_last_one(dummy_input)

#     print("Input: ", dummy_input)
#     print("Second-to-last layer output: ", output1)
#     print("Last layer output:", output2)
#     print()

for j in range(1000): # Test suite of inputs to last layer
    dummy_input = torch.randn(1, 48).clamp(min=0) # Clamp test tensors b/c ReLU() means they will be nonnegative

    last_layer = model[-2]

    output = last_layer(dummy_input);

    print("Input to last layer:", dummy_input)
    print("Last layer output:", output)
    print()

