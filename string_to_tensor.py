import torch

def string_to_tensor(s: str) -> torch.Tensor:
    codepoints = [ord(c) for c in s[:55]]  # convert up to 55 characters
    codepoints += [0] * (55 - len(codepoints))  # pad with zeros if needed
    return torch.tensor([codepoints], dtype=torch.float32)