import torch
import torch.nn as nn

# Load the model
model = torch.load("models/model.pt", weights_only=False)
model.eval()

def trace_connections(model, start_neuron_idx=0):
    """
    Trace through the model starting from the first input neuron (index 0)
    and find all neurons that have nonzero connections.
    """
    # Start with the first input neuron
    active_neurons = {0: [start_neuron_idx]}  # layer_idx -> list of active neuron indices
    layer_idx = 0
    
    # Trace through each layer
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            current_active = active_neurons.get(layer_idx, [])
            next_active = []
            
            # For each active neuron in current layer, find neurons in next layer with nonzero weights
            for neuron_idx in current_active:
                if neuron_idx < layer.weight.shape[0]:  # Check if neuron exists in this layer
                    # Find neurons in next layer with nonzero weights from this neuron
                    weights_from_neuron = layer.weight[:, neuron_idx]
                    nonzero_indices = torch.nonzero(weights_from_neuron).flatten()
                    next_active.extend(nonzero_indices.tolist())
            
            # Remove duplicates and sort
            next_active = sorted(list(set(next_active)))
            layer_idx += 1
            active_neurons[layer_idx] = next_active
            
            print(f"Layer {i}: {len(current_active)} active neurons -> {len(next_active)} active neurons")
    
    return active_neurons

def create_smaller_model(model, active_neurons):
    """
    Create a smaller model using only the active neurons and their connections.
    """
    layers = []
    linear_layer_idx = 0
    
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            current_active = active_neurons.get(linear_layer_idx, [])
            next_active = active_neurons.get(linear_layer_idx + 1, [])
            
            print(f"Creating layer {i}: {len(current_active)} -> {len(next_active)} neurons")
            
            if linear_layer_idx == 0:
                # First layer: only take first input and connect to active neurons
                in_features = 1  # Only first character
                out_features = len(next_active)
                
                # Create weight matrix: only connections from first input to active neurons
                new_weight = torch.zeros(out_features, in_features)
                for j, neuron_idx in enumerate(next_active):
                    if neuron_idx < layer.weight.shape[0]:
                        new_weight[j, 0] = layer.weight[neuron_idx, 0]  # Connection from first input
                
                # Create bias: only for active neurons
                new_bias = torch.zeros(out_features)
                for j, neuron_idx in enumerate(next_active):
                    if neuron_idx < layer.bias.shape[0]:
                        new_bias[j] = layer.bias[neuron_idx]
                
                new_layer = nn.Linear(in_features, out_features)
                new_layer.weight.data = new_weight
                new_layer.bias.data = new_bias
                layers.append(new_layer)
                
            else:
                # Hidden layers: connect active neurons from previous layer to active neurons in current layer
                in_features = len(current_active)
                out_features = len(next_active)
                
                # Create weight matrix: only connections between active neurons
                new_weight = torch.zeros(out_features, in_features)
                for j, out_neuron_idx in enumerate(next_active):
                    for k, in_neuron_idx in enumerate(current_active):
                        if (out_neuron_idx < layer.weight.shape[0] and 
                            in_neuron_idx < layer.weight.shape[1]):
                            new_weight[j, k] = layer.weight[out_neuron_idx, in_neuron_idx]
                
                # Create bias: only for active neurons
                new_bias = torch.zeros(out_features)
                for j, neuron_idx in enumerate(next_active):
                    if neuron_idx < layer.bias.shape[0]:
                        new_bias[j] = layer.bias[neuron_idx]
                
                new_layer = nn.Linear(in_features, out_features)
                new_layer.weight.data = new_weight
                new_layer.bias.data = new_bias
                layers.append(new_layer)
            
            linear_layer_idx += 1
            
        elif isinstance(layer, nn.ReLU):
            # Keep ReLU layers
            layers.append(layer)
    
    return nn.Sequential(*layers)

# Trace connections starting from first input neuron
print("Tracing connections from first input neuron...")
active_neurons = trace_connections(model)

# Create smaller model
print("\nCreating smaller model...")
smaller_model = create_smaller_model(model, active_neurons)

# Save the smaller model
torch.save(smaller_model, "models/first_char_model.pt")
print(f"Smaller model saved as 'models/first_char_model.pt'")

# Print model summary
print(f"\nOriginal model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Smaller model parameters: {sum(p.numel() for p in smaller_model.parameters()):,}")

# Test the smaller model
test_input = torch.tensor([[1.0]], dtype=torch.float32)  # First character only
with torch.no_grad():
    output = smaller_model(test_input)
    print(f"Test output shape: {output.shape}")
    print(f"Test output: {output}")