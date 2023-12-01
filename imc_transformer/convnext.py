import timm
import torch
from torch import nn


def DepthwiseConv2d(dim: int, kernel_size=7, padding=3):
    """
    Depthwise convolutional block (in_channel = out_channel), see: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L28
    :param dim: The channels.
    :param kernel_size: The kernel size.
    :param padding: The padding.
    :return: The module.
    """
    return nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)


def PointwiseConv2d(dim: int, out_dim: int):
    """
    Pointwise convolutional block, see https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L30
    :param dim: The channels.
    :param out_dim: The output channels.
    :return: The module.
    """
    # Note the paper describes this as a 1x1 convolution, which is equivalent to a linear layer
    return nn.Linear(dim, out_dim)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block as described in https://arxiv.org/abs/2201.03545
    """

    def __init__(
        self,
        in_channels: int,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
        dropout: float = 0,
    ):
        super().__init__()
        self.dconv = DepthwiseConv2d(in_channels)
        self.layer_norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.channel_last = ChannelLastConverter()
        self.pwconv = PointwiseConv2d(in_channels, in_channels * expansion)
        self.activation = nn.GELU()
        self.pwconv2 = PointwiseConv2d(in_channels * expansion, in_channels)
        self.layer_scale = nn.Parameter(
            torch.ones(1) * layer_scale_init, requires_grad=True
        )
        self.channel_first = ChannelFirstConverter()
        self.drop_path = timm.layers.DropPath(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x_orig = x
        x = self.dconv(x)
        # Move channels to end
        x = self.channel_last(x)
        x = self.layer_norm(x)
        x = self.pwconv(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        x = self.layer_scale * x
        # Move channels back to front
        x = self.channel_first(x)
        x = x_orig + self.drop_path(x)
        return x


class ChannelLastConverter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class ChannelFirstConverter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt as described in https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L52"""

    def __init__(
        self,
        channels: int,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        dropout: float = 0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()

        # Stem is the layer which does the initial convolution
        stem = nn.Sequential(
            nn.Conv2d(channels, dims[0], kernel_size=4, stride=4),
            # Need to move channels to end for layer norm
            ChannelLastConverter(),
            nn.LayerNorm(dims[0], eps=1e-6),
            # Move channels back to front
            ChannelFirstConverter(),
        )
        self.downsampling = nn.ModuleList([stem])
        for i in range(len(dims) - 1):
            self.downsampling.append(
                nn.Sequential(
                    ChannelLastConverter(),
                    nn.LayerNorm(dims[i], eps=1e-6),
                    ChannelFirstConverter(),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.feature_stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, dropout, sum(depths))]
        curr = 0
        for i in range(len(dims)):
            self.feature_stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dims[i],
                            layer_scale_init=layer_scale_init,
                            dropout=dp_rates[curr + j],
                        )
                        for j in range(depths[i])
                    ]
                )
            )
            curr += depths[i]

            self.final_norm = nn.Sequential(
                ChannelLastConverter(),
                nn.LayerNorm(dims[-1], eps=1e-6),
                ChannelFirstConverter(),
            )
            # self.head = nn.Conv2d(dims[-1], out_channels, kernel_size=1)

    def forward(self, x):
        features = []
        for i in range(len(self.downsampling)):
            x = self.downsampling[i](x)
            x = self.feature_stages[i](x)
            features.append(x)

        x = self.final_norm(x)
        # return self.head(self.final_norm(x))
        return x, features


class BasicConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),  # TODO: Would layer norm be better?
            nn.GELU(),
            # Final 1x1 convolution to get to the desired number of channels
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
        )

    def forward(self, x):
        return self.conv(x)


class ConvNeXtDecoder(nn.Module):
    """
    Inspired by Zhang, Hongbin, et al. "BCU-Net: Bridging ConvNeXt and U-Net for medical image segmentation." Computers in Biology and Medicine 159 (2023): 106960.
    This completes the u-net architecture by progressively upsampling with residual connections just like a U-Net.

    """

    def __init__(
        self,
        out_channels: int,
        depths: list[int] = [3, 9, 3, 3],
        dims: list[int] = [768, 384, 192, 96],
        dropout: float = 0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        # Double the dimensions for all, except the last step, that needs to be quadrupeled
        self.up_samples = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                )
                for i in range(len(dims) - 1)
            ]
        )

        self.decoders = nn.ModuleList(
            [BasicConvBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        # Must up-sample the stem
        self.final_upsample = nn.ConvTranspose2d(
            dims[-1],
            dims[-1],
            kernel_size=4,
            stride=4,
        )
        self.final_decoder = BasicConvBlock(dims[-1], dims[-1])

    def forward(self, x, compressed_features):
        for i, (features, deconv, decoder_block) in enumerate(
            zip(compressed_features, self.up_samples, self.decoders)
        ):
            x = deconv(x)
            x = torch.nn.functional.pad(
                x,
                (
                    0,
                    features.shape[3] - x.shape[3],
                    0,
                    features.shape[2] - x.shape[2],
                ),
            )
            x = torch.cat((x, features), dim=1)
            x = decoder_block(x)

        x = self.final_upsample(x)
        x = self.final_decoder(x)
        return x


class ConvNeXtUnet(nn.Module):
    """
    ConvNeXt
    Modified to accommodate image2image tasks via a basic U-net architecture.
    Inspired by: Zhang, Hongbin, et al. "BCU-Net: Bridging ConvNeXt and U-Net for medical image segmentation." Computers in Biology and Medicine 159 (2023): 106960.
    Idea: The "encoder" part of the model is the typical ConvNeXt architecture, but it is followed by a U-Net-like decoder.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        dropout: float = 0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            channels, depths, dims, dropout, layer_scale_init
        )
        self.decoder = ConvNeXtDecoder(
            out_channels,
            list(reversed(depths)),
            list(reversed(dims)),
            dropout,
            layer_scale_init,
        )
        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = torch.log1p(x)
        x, features = self.encoder(x)
        x = self.decoder(x, features[::-1][1:])
        x = self.head(x)
        x = torch.expm1(x)
        return x
