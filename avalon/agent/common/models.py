from enum import Enum
from typing import Callable
from typing import Dict
from typing import List

import torch
from torch import Tensor


class ActivationFunction(Enum):
    ACTIVATION_EXP = "ACTIVATION_EXP"
    ACTIVATION_SOFTPLUS = "ACTIVATION_SOFTPLUS"
    ACTIVATION_ELU = "ACTIVATION_ELU"
    ACTIVATION_RELU = "ACTIVATION_RELU"
    ACTIVATION_LEAKY_RELU = "ACTIVATION_LEAKY_RELU"
    ACTIVATION_TANH = "ACTIVATION_TANH"


# maps from str instead of function so this even works with reloading
ACTIVATION_MODULE_LOOKUP: Dict[str, Callable[[], torch.nn.Module]] = {
    ActivationFunction.ACTIVATION_SOFTPLUS.value: torch.nn.Softplus,
    ActivationFunction.ACTIVATION_ELU.value: torch.nn.ELU,
    ActivationFunction.ACTIVATION_RELU.value: torch.nn.ReLU,
    ActivationFunction.ACTIVATION_LEAKY_RELU.value: torch.nn.LeakyReLU,
    ActivationFunction.ACTIVATION_TANH.value: torch.nn.Tanh,
}


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation_function: ActivationFunction = ActivationFunction.ACTIVATION_RELU,
        is_batch_normalized: bool = False,
    ) -> None:
        """Will have `num_layers` `torch.nn.Linear` layers.

        So num_layers=2 will yield a MLP with one hidden layer and one output layer."""
        super().__init__()
        self.act_fn = ACTIVATION_MODULE_LOOKUP[activation_function.value]

        assert num_layers >= 0, "negative layers?!?"
        if num_layers == 0:
            assert input_dim == output_dim, "For identity MLP, input dim must equal output dim"
            self.net: torch.nn.Module = torch.nn.Identity()
            return

        if num_layers == 1:
            self.net = torch.nn.Linear(input_dim, output_dim)
            return

        layers: List[torch.nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(self.act_fn())
            if is_batch_normalized:
                layers.append(torch.nn.BatchNorm1d(hidden_dim))

            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        result = self.net(x)  # type: ignore
        assert isinstance(result, Tensor)
        return result
