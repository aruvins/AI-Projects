import torch


def positional_encoding(x, L=10):
    encoding = [x]

    for i in range(L):
        encoding.append(torch.sin((2.0 ** i) * x))
        encoding.append(torch.cos((2.0 ** i) * x))

    return torch.cat(encoding, dim=-1)