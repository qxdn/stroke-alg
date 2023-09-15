import torch


def load_weight(model,weight_path):
    model.load_state_dict(torch.load(weight_path))
    return model