import torch

# Load the model
model = torch.load("models/model.pt", weights_only=False)
model.eval()

# Hook to capture output of second-to-last layer (before final ReLU)
captured = {}

def hook_fn(module, input, output):
    captured['pre_final'] = output

# Register hook on the final Linear layer (just before last ReLU)
hook = model[-2].register_forward_hook(hook_fn)

# Input file
input_file = "two_word_permutations.txt"

# Process phrases
with open(input_file, "r") as f:
    for line in f:
        phrase = line.strip()
        if not phrase:
            continue
        try:
            with torch.no_grad():
                _ = model(phrase)  # triggers hook
                vec = captured['pre_final']
                print(f"{phrase}: {vec.squeeze().tolist()}")
        except Exception as e:
            print(f"Error processing '{phrase}': {e}")

hook.remove()