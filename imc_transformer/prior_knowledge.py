import torch
from torch import nn
import torch.nn.functional as F


# Ref: https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
class ChannelAttentionSubModule(nn.Module):
    def __init__(self, channels: int, reduction_ratio=2):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
        )

    def forward(self, x, attention_bias=None):
        """
        Apply channel attention to the input.
        :return: Attended input.
        """
        avg_pool = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        max_pool = F.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        original_attn = self.mlp(avg_pool) + self.mlp(max_pool)

        if attention_bias is not None:
            original_attn = original_attn * attention_bias

        channel_attn = F.sigmoid(original_attn).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * channel_attn


class PriorKnowledgeChannelAttention(nn.Module):
    """
    This is an Attention module that can incorporate the known encoding prior knowledge.
    Since we know, a priori, which compressed channels correspond to uncompressed channels, we can
    use this information to guide the attention across compressed channels.

    This is based on the Channel Attention module from the paper "CBAM: Convolutional Block Attention Module" by
    Woo et al. (https://arxiv.org/abs/1807.06521).
    """

    def __init__(
        self, in_channels: int, codebook: torch.Tensor = None, reduction_ratio=2
    ):
        super().__init__()
        # Codebook requires no gradients
        self.register_buffer("codebook", codebook)
        if codebook is not None:
            self.codebook.requires_grad_(False)
            self.prior_knowledge_attention = nn.Linear(in_channels, codebook.shape[0])
        else:
            self.prior_knowledge_attention = lambda x: None  # Get no map
        self.channel_attention = ChannelAttentionSubModule(in_channels, reduction_ratio)

    def forward(self, f):
        return self.channel_attention(f, self.prior_knowledge_attention(f))


# Authors of CBAM show spatial attention adds additional performance gains
# Ref: https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L72
class SpatialAttentionSubModule(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        channel_max_pooling = lambda x: x.max(dim=1)[0].unsqueeze(1)
        channel_avg_pooling = lambda x: x.mean(dim=1).unsqueeze(1)
        self.channel_pooling = lambda x: torch.cat(
            (channel_max_pooling(x), channel_avg_pooling(x)), dim=1
        )
        # Args are from the CBAM implementation
        self.conv = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                dilation=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, x):
        pooled = self.channel_pooling(x)
        spatial_attn = self.conv(pooled)
        return x * F.sigmoid(spatial_attn)


class PriorKnowledgeCBAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        codebook: torch.Tensor = None,
        reduction_ratio: int = 2,
        spatial: bool = True,
        channel: bool = True,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attention = (
            PriorKnowledgeChannelAttention(in_channels, codebook, reduction_ratio)
            if channel
            else nn.Identity()
        )
        self.spatial_attention = (
            SpatialAttentionSubModule(kernel_size) if spatial else nn.Identity()
        )

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))
