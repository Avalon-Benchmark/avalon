import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from avalon.common.log_utils import logger


def forward_activation_hook(module: torch.nn.Module, forward_input: Tensor, forward_output: Tensor):
    if hasattr(module, "bias") and module.bias is not None:
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module._output_mean = forward_output.mean(dim=(0, 2, 3), keepdim=True)
        elif isinstance(module, torch.nn.Linear):
            module._output_mean = forward_output.mean(dim=0)
        else:
            raise NotImplementedError
        forward_output = forward_output - module._output_mean

    module._output_std = forward_output.std()
    forward_output = forward_output / module._output_std

    return forward_output


def module_pre_initialization(module: torch.nn.Module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        module._hook_handle = module.register_forward_hook(forward_activation_hook)  # type: ignore


@torch.no_grad()
def update_initialization(module: torch.nn.Module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        if hasattr(module, "_output_mean"):
            assert module.bias is not None
            assert isinstance(module._output_mean, Tensor)
            assert isinstance(module._output_std, Tensor)
            module.bias.add_(-module._output_mean.squeeze() / module._output_std)
            del module._output_mean

        if hasattr(module, "_output_std"):
            assert module.weight is not None
            assert isinstance(module._output_std, Tensor)
            module.weight.mul_(1.0 / module._output_std)
            del module._output_std

        module._hook_handle.remove()  # type: ignore
        del module._hook_handle


def apply_data_dependent_init(method: LightningModule):
    # TODO: this crashes silently when you run out of cuda memory, not sure how to properly check or fix this yet
    logger.warning(
        "WARNING: using data dependent initialization. This crashes silently when you run out of cuda memory. Be aware when using it."
    )
    method.apply(module_pre_initialization)
    method.training_step(next(iter(method.train_dataloader())), batch_idx=0)
    method.apply(update_initialization)
