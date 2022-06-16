import torch
from torch import Tensor
from torch import nn


def test_forward_inference() -> Tensor:
    """Strange. We're implicitly using this file to test the plugin, because it won't type check without the plugin..."""
    layer = nn.Linear(16, 16)
    x = torch.rand((16,))
    y = layer(x)
    return y
