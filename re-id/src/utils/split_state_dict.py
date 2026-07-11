import torch
import os


def save_state_dict(state_dict, save_dir: str, model_name: str, max_size_mb=90):
    parts = []
    current = {}
    current_size = 0

    max_size = max_size_mb * 1024 * 1024

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current and current_size + tensor_size > max_size:
            parts.append(current)
            current = {}
            current_size = 0

        current[key] = tensor
        current_size += tensor_size

    if current:
        parts.append(current)

    if len(parts) == 1:
        torch.save(parts[0], os.path.join(save_dir, f"{model_name}.pt"))
    else:
        for i, part in enumerate(parts):
            part_path = os.path.join(save_dir, f"{model_name}_part_{i}.pt")
            torch.save(part, part_path)


def load_state_dict(save_dir: str, model_name: str):
    state_dict = {}
    i = 0
    while True:
        part_path = os.path.join(save_dir, f"{model_name}_part_{i}.pt")
        try:
            part = torch.load(part_path)
            state_dict.update(part)
            i += 1
        except FileNotFoundError:
            break

    if i == 0:  # No parts found, try loading the whole state dict
        try:
            part_path = os.path.join(save_dir, f"{model_name}.pt")
            state_dict = torch.load(part_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No state dict found for model '{model_name}' in directory '{save_dir}'.")

    return state_dict
