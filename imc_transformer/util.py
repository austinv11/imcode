import functools
import math

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
