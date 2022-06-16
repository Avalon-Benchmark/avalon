import torch
from torch import Tensor
from torch.nn.functional import grid_sample

from common.imports import np
from common.utils import T


def noop(x: T) -> T:
    return x


def swap_channels_for_pytorch(tensor: Tensor) -> Tensor:
    # pytorch expects CxHxW
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    elif len(tensor.shape) == 4:
        tensor = tensor.permute(0, 3, 1, 2)
    elif len(tensor.shape) == 5:
        tensor = tensor.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(f"Unexpected number of dimensions {tensor.shape}, expected 3 or 4 dimensions.")
    return tensor


def _tensor_to_float32(tensor: Tensor) -> Tensor:
    return tensor.float().type(torch.float32)


def convert_array_to_tensor(array: np.ndarray) -> Tensor:
    return torch.from_numpy(array).contiguous()


def rescale_lab(tensor: Tensor) -> Tensor:
    """Transform LAB floats in (~-100, ~100) to (0, 1)"""
    return torch.clamp((tensor + 100) / 200, 0, 1)


def unrescale_lab(tensor: Tensor) -> Tensor:
    """Transform LAB floats in (0, 1) to (~-100, ~100)"""
    return (tensor * 200) - 100


def rgb_float_to_rgb_int(x: Tensor) -> Tensor:
    """Converts RGB floats in the range of [0, 1] to uint8s in [0, 255]. Assumes CxHxW"""
    assert x.shape[0] == 3
    assert len(x.shape) == 3
    return (x[:, ...] * 255).type(torch.uint8)


def tanh_to_rgb_int(x: Tensor) -> Tensor:
    """Converts logits in [-1, 1] to uint8s in [0, 255]. Assumes CxHxW"""
    assert x.shape[0] == 3
    assert len(x.shape) == 3
    return ((x + 1) / 2 * 255).type(torch.uint8)


def coord_conv(height: int, width: int, is_video: bool) -> Tensor:
    x = np.linspace(-1, 1, height)
    y = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)
    tensor = torch.stack([torch.FloatTensor(xx), torch.FloatTensor(yy)], dim=0).unsqueeze(0)
    return tensor.unsqueeze(0) if is_video else tensor


class SwapChannels(torch.nn.Module):
    def __init__(self, feature: str):
        super().__init__()
        # see TASK: df78d537-7ed3-4428-8e5f-44a1a16034e7
        self.feature = feature

    def forward(self, tensor: Tensor) -> Tensor:
        # TASK: df78d537-7ed3-4428-8e5f-44a1a16034e7 - sigh ... depth doesn't output a dimension for channels
        #  this is a hacky work around that makes sure we have the right shape
        if self.feature == "depth":
            tensor = tensor.unsqueeze(-1)
        return swap_channels_for_pytorch(tensor)


class RGBByteToFloat(torch.nn.Module):
    """
    Converts a tensor in the range 0...255 to 0.0...1.0.

    Note: this actually expects a float as input.
    """

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = _tensor_to_float32(tensor)
        return tensor.div(255)


class RescaleDepth(torch.nn.Module):
    def __init__(self, is_converting_to_float32: bool = True):
        super(RescaleDepth, self).__init__()
        self.is_converting_to_float32 = is_converting_to_float32

    def forward(self, tensor: Tensor) -> Tensor:
        # if tensor.max() > 200:
        #     raise NotImplementedError(
        #         """We're squashing each frame's depth features with a sigmoidal function (see here for graph: https://www.geogebra.org/calculator/upecbrb5),
        #         but vanilla sigmoid scales things too dramatically - everything beyond 200 meters is basically considered "max distance".
        #         Remove this assert if this no longer applies."""
        #     )
        if self.is_converting_to_float32:
            tensor = _tensor_to_float32(tensor)
        return 2 * torch.exp(tensor / 5) / (torch.exp(tensor / 5) + 1) - 1


# TODO pitch and yaw should work over the batch
def crop_360_video(
    video: Tensor,  # (bs, sl, c, h, w)
    pitch: Tensor,  # (bs, sl,)
    yaw: Tensor,  # (bs, sl,)
    fov_x: int = 90,
    fov_y: int = 90,
    old_fov_x: int = 360,
    old_fov_y: int = 180,
) -> Tensor:
    batch_size, sequence_length, channels, height, width = video.shape
    pixel_center_x = int(yaw / old_fov_x * width)
    pixel_center_y = int(pitch / old_fov_y * height)
    pixel_center_position = (
        torch.tensor([pixel_center_x, pixel_center_y], device=video.device).unsqueeze(0).repeat(batch_size, 1, 1)
    )

    # grid_sample expects values between -1 and 1 with (0, 0) being the center of the image
    corrected_center_position = torch.zeros_like(pixel_center_position).float()
    corrected_center_position[..., -1] = (pixel_center_position[..., -1] / height - 0.5) / 0.5
    corrected_center_position[..., -2] = ((width - pixel_center_position[..., -2]) / width - 0.5) / 0.5

    # calculates the new dimensions of the image using the FOV
    new_pixel_width = int(fov_x / old_fov_x * width)
    new_pixel_height = int(fov_y / old_fov_y * height)

    # pixel values of the patch offsets
    pixel_x_offsets = torch.arange(0, new_pixel_width, dtype=torch.float32, device=video.device)
    pixel_y_offsets = torch.arange(0, new_pixel_height, dtype=torch.float32, device=video.device)

    # pixel locations of patch in x, y converted to be within the range [-1, 1] (corrected range)
    corrected_x_offsets = (pixel_y_offsets / (new_pixel_height - 1) - 0.5) / 0.5 * (new_pixel_height / width)
    corrected_y_offsets = (pixel_x_offsets / (new_pixel_width - 1) - 0.5) / 0.5 * (new_pixel_width / height)
    # copies y along the columns, copies x along the rows
    corrected_grid_y, corrected_grid_x = torch.meshgrid(corrected_y_offsets, corrected_x_offsets)
    # creates a grid where each row the values of x increases sequentially and each column the values of y increases
    #   sequentially
    corrected_offsets = (
        torch.stack([corrected_grid_x, corrected_grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    )

    centers = corrected_center_position.view(batch_size, 1, 1, 2).repeat(1, new_pixel_height, new_pixel_width, 1)
    corrected_points = corrected_offsets + centers

    cropped_frames = []
    for t in range(sequence_length):
        frame = video[:, t]
        # shifted by +1 and back so black pixels will actually be black (assuming video input is between -1 and 1)
        cropped_frame = grid_sample(frame + 1.0, corrected_points, align_corners=True) - 1.0
        cropped_frames.append(cropped_frame)
    return torch.stack(cropped_frames, dim=1)
