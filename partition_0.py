import torch
import torch.nn as nn

# Load the first character model
model = torch.load("models/first_char_model.pt", weights_only=False)
model.eval()

# Create a hook to capture the second-to-last layer output
second_to_last_output = None

def hook_fn(module, input, output):
    global second_to_last_output
    second_to_last_output = output

# Register hook on the last linear layer (before final ReLU)
linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
if len(linear_layers) >= 1:
    last_linear = linear_layers[-1]
    last_linear.register_forward_hook(hook_fn)
    print(f"Registered hook on last linear layer: {last_linear}")
else:
    print("Model doesn't have any linear layers")
    exit()

print("Running first character model on integers 0-128...")
print("Input | Last linear layer output (before ReLU)")

for i in range(154999):  # 0 to 128 inclusive
    # Create input tensor with the integer value
    input_tensor = torch.tensor([[float(i)]], dtype=torch.float32)
    
    # Run the model
    with torch.no_grad():
        output = model(input_tensor)
        
        # Print the last linear layer output (single number)
        if second_to_last_output is not None:
            output_value = second_to_last_output.item()
            print(f"{i:3d} | {output_value:.6f}")
        else:
            print(f"{i:3d} | No output captured")

print(f"\nTotal inputs processed: 129 (0-128)")
print(f"Last linear layer shape: {second_to_last_output.shape if second_to_last_output is not None else 'Unknown'}")
