import functools
import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


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
    n_channels: int, n_compressed_channels: int = None, spike_in: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random binary codebook that maps n_channels to n_compressed_channels.
    :param n_channels: Number of input channels.
    :param n_compressed_channels: Number of output channels.
    :param spike_in: The number of randomly assigned input channels to be mapped to additional totally uncompressed channels.
    :return: n_channels x n_compressed_channels codebook, and a list of spike-in channels.
    """
    if n_compressed_channels is None:
        n_compressed_channels = torch.ceil(torch.log2(torch.tensor(n_channels))).int()

    total_output_channels = n_compressed_channels + spike_in
    # Generate a random codebook, where each input channel maps to two output channels.
    # And additionally attempt to evenly distribute the channels across the output channels.
    codebook = torch.randint(
        0, n_compressed_channels, (n_channels, 2), dtype=torch.long
    )
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
def compress(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Given images that are C x H x W, compress them based on a codebook that is C x C_compressed.
    :param x: N x C x H x W images.
    :param codebook: C x C_compressed channel mapping.
    :return: N x C_compressed x H x W compressed images.
    """
    # Channel first to channel last.
    x = x.permute(0, 2, 3, 1)
    # Compress the channels.
    x = x @ codebook
    # Channel last to channel first.
    x = x.permute(0, 3, 1, 2)
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
