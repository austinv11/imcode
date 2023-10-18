import itertools
import random

import torch
import tifffile as tiff
import matplotlib.pyplot as plt


def read_tiff(path: str) -> torch.tensor:
    dat = tiff.imread(path)
    return torch.tensor(dat)


def percentile_clip(
    data: torch.tensor, percentile: float, axis: int = 0
) -> torch.tensor:
    """
    Clip the data along an axis to a given maximum percentile value.
    :param data: The data.
    :param percentile: The percentile value to clip to.
    :return: The clipped data.
    """
    max_val = torch.quantile(data, percentile, dim=axis, keepdim=True)
    return torch.minimum(data, max_val)


def reconstruct_linear(
    compressed_data: torch.Tensor, compressed_assignments: torch.Tensor
) -> torch.Tensor:
    # Given compressed data, X, and assignments A, we want to solve the system of equations X = A * S, where S is the
    # original data.
    # Now, we can solve the system of equations
    solution = torch.linalg.lstsq(
        compressed_assignments.float().T, compressed_data.flatten(1)
    ).solution.reshape(compressed_assignments.shape[0], *compressed_data.shape[1:])
    # Scale the solution based on compressed_data range
    solution = solution * compressed_data.max()
    return solution


def main(file: str):
    print("Reading file: ", file)
    data = read_tiff(file)
    print("Data shape: ", data.shape)

    print(
        "Clipping data to 98th percentile to remove outliers"
    )  # TODO: Check if we need this
    data = percentile_clip(data, 0.98)
    plt.imshow(torch.log1p(data.sum(axis=0)))
    plt.show()
    plt.imshow(torch.log1p(data[1, :, :]))
    plt.show()

    print("Compressing the data")
    compressed_data, compressed_assignments = synthetic_compression(data)
    print("Compressed data shape: ", compressed_data.shape)
    print("Compressed assignments shape: ", compressed_assignments.shape)
    plt.imshow(torch.log1p(compressed_data.sum(axis=0)))
    plt.show()
    plt.imshow(torch.log1p(compressed_data[1, :, :]))
    plt.show()

    print("Data compressed, attempting to reconstruct")
    reconstructed_data = reconstruct_linear(compressed_data, compressed_assignments)
    # Peek at the first 6 channels from original, compressed, and reconstructed data
    fig, axes = plt.subplots(3, 6, figsize=(30, 30))
    for i in range(6):
        axes[0, i].imshow(data[i, :, :].log1p())
        axes[0, i].set_title(f"Original Channel {i + 1}")
        axes[1, i].imshow(compressed_data[i, :, :].log1p())
        axes[1, i].set_title(f"Compressed Channel {i + 1}")
        axes[2, i].imshow(reconstructed_data[i, :, :].log1p())
        axes[2, i].set_title(f"Reconstructed Channel {i + 1}")
    plt.tight_layout()
    plt.savefig("plots/compressed-sensing.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # main("raw/TMA1_45B.tiff")
    from imc_transformer import model

    model.main()
