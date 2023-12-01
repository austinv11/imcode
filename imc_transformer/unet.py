import torch
import torch.nn as nn
import torchvision

from prior_knowledge import PriorKnowledgeCBAM


# Implementation inspired by https://amaarora.github.io/posts/2020-09-13-unet.html#u-net
class ConvBlock(nn.Module):
    """
    Basic convolutional building block for the U-Net.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chans, out_chans, kernel_size, padding=padding, stride=stride
        )
        self.conv2 = nn.Conv2d(
            out_chans, out_chans, kernel_size, padding=padding, stride=stride
        )
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(out_chans)
        self.batch_norm2 = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        return self.relu(
            self.batch_norm2(self.conv2(self.relu(self.batch_norm1(self.conv1(x)))))
        )


class ContractiveBlock(nn.Module):
    """
    Contractive block for the U-Net.
    """

    def __init__(
        self,
        channels: tuple[int, ...],
        kernel_size: int = 3,
        padding: int = 1,
        attention: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvBlock(channels[i], channels[i + 1], kernel_size, padding)
                for i in range(len(channels) - 1)
            ]
        )
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size - 1)
        if attention:
            self.attention = nn.ModuleList(
                [PriorKnowledgeCBAM(channels[i]) for i in range(len(channels))]
            )
        else:
            self.attention = None

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.blocks):
            if self.attention is not None:
                x = self.attention[i](x)
            x = block(x)
            feats.append(x)
            x = self.pooling(x)
        return feats


class ExpansiveBlock(nn.Module):
    """
    Up-convolution of the image.
    """

    def __init__(
        self,
        channels: tuple[int, ...],
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        attention: bool = False,
    ):
        super().__init__()
        self.deconvolutions = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size,
                    padding=padding,
                    stride=stride,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                ConvBlock(
                    channels[i], channels[i + 1], kernel_size + 1, padding + 1, stride
                )
                for i in range(len(channels) - 1)
            ]
        )
        if attention:
            self.attention = nn.ModuleList(
                [PriorKnowledgeCBAM(channels[i + 1]) for i in range(len(channels) - 1)]
            )
        else:
            self.attention = None

    def forward(self, x, contractive_features):
        for i, (features, deconv, decoder_block) in enumerate(
            zip(contractive_features, self.deconvolutions, self.decoder_blocks)
        ):
            x = deconv(x)
            # Pad the input to match the encoder features rather than cropping
            x = torch.nn.functional.pad(
                x,
                (
                    0,
                    features.shape[3] - x.shape[3],
                    0,
                    features.shape[2] - x.shape[2],
                ),
            )
            if self.attention is not None:
                x = self.attention[i](x)
            x = torch.cat([x, features], dim=1)
            x = decoder_block(x)
        return x


class UNet(nn.Module):
    """
    Basic U-Net implementation.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        image_size: tuple[int, int],
        attention: bool = False,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.image_size = image_size

        self.encoder = ContractiveBlock(
            (in_chans, 64, 128, 256, 512, 1024),
            kernel_size=3,
            padding=1,
            attention=attention,
        )
        self.decoder = ExpansiveBlock(
            (1024, 512, 256, 128, 64), kernel_size=2, padding=0, attention=attention
        )
        self.head = nn.Conv2d(64, out_chans, kernel_size=1)

    def forward(self, x):
        # Input data is counts, take the log1p to make it easier for the model to learn
        x = torch.log1p(x)
        feats = self.encoder(x)
        x = self.decoder(feats[-1], feats[::-1][1:])
        x = self.head(x)
        x = torch.relu(x)  # Must be positive
        # The target data is counts, and we took the log1p of the inputs, so we need to undo that and round
        x = torch.expm1(x)
        return x
