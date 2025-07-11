import torch

# Load the model
model = torch.load("models/model.pt", weights_only=False)
model.eval()

# Hook to capture the input to the first layer
captured = {}

def hook_fn(module, input, output):
    captured['in'] = input[0]  # input is a tuple

# Register the hook on the first layer
hook = model[0].register_forward_hook(hook_fn)

# Input file
input_file = "ascii_sequences_non_overlap_no_pad_hex.txt"

# Process phrases
with open(input_file, "r") as f:
    for line in f:
        phrase = line.strip()
        if not phrase:
            continue
        try:
            with torch.no_grad():
                _ = model(phrase)  # triggers hook
                input_tensor = captured['in']
                print(f"{phrase}: {input_tensor.squeeze().tolist()}")
        except Exception as e:
            print(f"Error processing '{phrase}': {e}")

hook.remove()