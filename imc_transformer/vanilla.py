import torch.nn as nn
import torch


class VanillaCNNModule(nn.Module):
    """
    This is a basic CNN that given an input image window, will output the values for the center pixel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        window_size: int = 32,
    ):
        super().__init__()
        # CNN that will be projected into a latent space, where a MLP will predict the center pixel
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        final_resolution = window_size // 4

        self.mlp = nn.Sequential(
            nn.Linear(64 * final_resolution * final_resolution, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, out_channels),
        )

    def forward(self, x):
        x = torch.log1p(x)
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        x = torch.expm1(x)
        return x


class CollatedVanillaCNN(nn.Module):
    """
    Given an input image, run the Vanilla CNN on all windows and collate the outputs to
    form a complete image2image prediction at the same resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        window_size: int = 4,
    ):
        super().__init__()
        self.cnn = VanillaCNNModule(in_channels, out_channels, kernel_size, window_size)
        self.window_size = window_size
        self.out_channels = out_channels

    def forward(self, x):
        # Holder for output
        output = x.new_zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]))
        # For each pixel, extract the window around it and run the CNN
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                # Extract window and pad if necessary
                window = x[:, :, i : i + self.window_size, j : j + self.window_size]
                if window.shape[2] < self.window_size:
                    window = nn.functional.pad(
                        window,
                        (0, 0, 0, self.window_size - window.shape[2]),
                        mode="constant",
                        value=0,
                    )
                if window.shape[3] < self.window_size:
                    window = nn.functional.pad(
                        window,
                        (0, self.window_size - window.shape[3], 0, 0),
                        mode="constant",
                        value=0,
                    )
                output[:, :, i, j] = self.cnn(window)
        return output
