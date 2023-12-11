import functools
import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


# Get the eps value for the device
EPS = torch.finfo(torch.float32).eps * 10


def _initialize(module: nn.Module, scale: float = 1.0, bias: bool = True):
    # https://github.com/tqch/poisson-jump/blob/d08e08b1e18e795538fbaf4abb3df5d8ca68120b/poisson_jump/nets/modules.py#L18C39-L18C69
    gain = math.sqrt(scale or 1e-10)
    default_initializer = functools.partial(nn.init.xavier_uniform_, gain=gain)
    match type(module):
        case nn.Linear | nn.Conv2d | nn.Conv1d | nn.ConvTranspose1d | nn.ConvTranspose2d:
            default_initializer(module.weight)
            if bias:
                nn.init.zeros_(module.bias)
        case nn.Sequential | nn.ModuleList:
            for m in module:
                _initialize(m, scale, bias)


@torch.jit.script
def random_codebook(
    n_channels: int, n_compressed_channels: int = -1, spike_in: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random binary codebook that maps n_channels to n_compressed_channels.
    :param n_channels: Number of input channels.
    :param n_compressed_channels: Number of output channels.
    :param spike_in: The number of randomly assigned input channels to be mapped to additional totally uncompressed channels.
    :return: n_channels x n_compressed_channels codebook, and a list of spike-in channels.
    """
    if n_compressed_channels < 1:  # If less than 1, then we need to generate it
        n_compressed_channels = torch.ceil(torch.log2(torch.tensor(n_channels))).int()

    total_output_channels = n_compressed_channels + spike_in
    # Generate a random codebook, where each input channel maps to two different output channels.
    codebook = torch.zeros((n_channels, 2), dtype=torch.long)
    for i in range(n_channels):
        codebook[i, :] = torch.randperm(n_compressed_channels)[:2]
    # Randomly select spike-in channels
    spike_in_channels = torch.randperm(n_channels)[:spike_in]
    # Add the spike-in channels to the codebook.
    codebook = torch.cat(
        (codebook, torch.zeros((n_channels, 1), dtype=torch.long)), dim=1
    )
    codebook[spike_in_channels, 2] = torch.arange(
        n_compressed_channels, total_output_channels
    )
    # Now we need to generate the binary mapping matrix (C x C_compressed).
    final_mapping = torch.zeros((n_channels, total_output_channels))
    for i in range(n_channels):
        for j in range(3):
            final_mapping[i, codebook[i, j]] = 1

    return final_mapping, spike_in_channels


@torch.jit.script
def mask_loss(
    elementwise_loss: torch.Tensor,
    mask: torch.Tensor,
    normalize: bool = True,
    reduce: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Take calculated elementwise loss and apply the mask to ignore missing data.
    :param elementwise_loss: The loss values per element (B x C x H x W).
    :param mask: The mask (B x H x W).
    :param normalize: If true, will normalize the loss by the number of unmasked elements.
    :param reduce: If true, will reduce the loss to a scalar by summing over the batch.
    :return: The masked loss.
    """
    mask = mask.detach().float()
    if elementwise_loss.ndim < 4:
        # Resize the mask
        mask = mask.unsqueeze(1)
    # Apply the mask
    masked_loss = elementwise_loss * mask

    dims = list(range(1, masked_loss.ndim))

    if reduce:
        masked_loss = masked_loss.sum(dim=dims)

    if normalize:
        masked_loss = (masked_loss + eps) / (torch.sum(mask, dim=dims) + eps)

    return masked_loss


@torch.jit.script
def compress(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Given images that are C x H x W, compress them based on a codebook that is C x C_compressed.
    :param x: N x C x H x W images.
    :param codebook: C x C_compressed channel mapping.
    :return: N x C_compressed x H x W compressed images.
    """
    if codebook.ndim < 3:  # Un-Batched codebook
        # Add a batch dimension to the codebook so that we can use batched matrix multiplication.
        codebook = codebook.unsqueeze(0)

    if x.ndim == 4:  # Batched input
        # # Channel first to channel last.
        # x = x.permute(0, 2, 3, 1)
        # # Compress the channels.
        # x = x @ codebook
        # # Channel last to channel first.
        # x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        if codebook.ndim < 3:  # Unbatched codebook
            # Add a batch dimension to the codebook so that we can use batched matrix multiplication.
            codebook = codebook.unsqueeze(0)
        # Reshape the input to be B x C x (H * W)
        x = x.reshape(B, C, -1)
        # Channel first to channel last.
        x = x.permute(0, 2, 1)
        # Compress the channels.
        x = x @ codebook
        # Channel last to channel first.
        x = x.permute(0, 2, 1)
        # Reshape the input to be B x C_compressed x H x W
        x = x.reshape(B, -1, H, W)
    elif x.ndim == 3:  # Non-batched input
        # Channel first to channel last.
        x = x.permute(1, 2, 0)
        # Compress the channels.
        x = x @ codebook
        # Channel last to channel first.
        x = x.permute(2, 0, 1)
    elif x.ndim == 2:  # Non-batched input, flattened image
        # Compress the channels.
        x = x @ codebook
    else:
        raise ValueError(f"Unsupported number of dimensions: {x.ndim}")
    return x


@torch.jit.script
def naive_uncompress(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Given images that are C x H x W, uncompress them based on a codebook that is C x C_compressed.
    :param x: N x C_compressed x H x W images.
    :param codebook: C x C_compressed channel mapping.
    :return: N x C x H x W uncompressed images.
    """
    # Channel first to channel last.
    x = x.permute(0, 2, 3, 1)
    # Uncompress the channels.
    x = x @ codebook.t()
    # Channel last to channel first.
    x = x.permute(0, 3, 1, 2)
    return x
