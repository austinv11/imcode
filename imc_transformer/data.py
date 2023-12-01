"""Generate the dataset.

Usage:
    data.py <dataset_path> <output_path> [--image_size=<image_size>] [--patch_overlap=<patch_overlap>]
    data.py (-h | --help)

Options:
    -h --help   Show this screen
    --image_size=<image_size>   The size to reshape the images to [default: None]
    --patch_overlap=<patch_overlap>   The amount of overlap between patches when resizing [default: 0]
"""

import itertools
import os.path
import sys
from glob import glob
from typing import Union, Optional
from pathlib import Path

import lightning.pytorch as pl
import torch
import torchvision
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    Compose,
    RandomErasing,
    RandomAffine,
)
from torch import nn, optim, utils
from torch.nn.utils.rnn import pad_sequence


def normalize_channel_ordering(
    curr_channel_labels: list[str],
    curr_tensors: list[torch.Tensor],
    new_channel_labels: list[str],
    new_tensor: torch.Tensor,
):
    if len(curr_channel_labels) == 0:  # No channels yet, just add the new ones
        return new_channel_labels, [new_tensor]

    # If the data has exactly matching channels, we can just add it to the list
    if all(
        [
            channel == curr_channel_labels[i]
            for i, channel in enumerate(new_channel_labels)
        ]
    ):
        channel_labels = curr_channel_labels
        tensors = curr_tensors + [new_tensor]
        return channel_labels, tensors

    # Append new channels, re-arrange new data, and backfill current data with zeros
    new_channels = [
        channel for channel in new_channel_labels if channel not in curr_channel_labels
    ]
    if len(new_channels) > 0:  # Need to backfill old data with zeros
        curr_channel_labels = curr_channel_labels + new_channels
        curr_tensors = [
            torch.cat([tensor, torch.zeros((len(new_channels), *tensor.shape[1:]))])
            for tensor in curr_tensors
        ]
    # Now we have to make the current data match the channel ordering and add missing channels as 0s
    new_data = torch.zeros((len(curr_channel_labels), *new_tensor.shape[1:]))
    for i, channel in enumerate(curr_channel_labels):
        added = False
        for j, file_channel in enumerate(new_channel_labels):
            if file_channel == channel:
                new_data[i, :, :] = new_tensor[j, :, :]
                added = True
                break
        if not added:
            new_data[i, :, :] = torch.zeros(new_tensor.shape[1:])
    curr_tensors.append(new_data)
    return curr_channel_labels, curr_tensors


def clear_empty_channels(
    curr_channel_labels: list[str], curr_tensors: list[torch.Tensor]
):
    all_zero_channel_candidates = list(curr_channel_labels)
    for dat in curr_tensors:
        for channel in tuple(all_zero_channel_candidates):
            if torch.any(dat[curr_channel_labels.index(channel), :, :] >= 1):
                all_zero_channel_candidates.remove(channel)
        if len(all_zero_channel_candidates) == 0:
            return curr_channel_labels, curr_tensors

    # Remove all zero channels
    for channel in all_zero_channel_candidates:
        channel_i = curr_channel_labels.index(channel)
        curr_channel_labels.pop(channel_i)
        for i, dat in enumerate(tuple(curr_tensors)):
            curr_tensors[i] = torch.cat(
                [dat[:channel_i, :, :], dat[channel_i + 1 :, :, :]]
            )
    return curr_channel_labels, curr_tensors


def safe_stack(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    torch.stack equivalent that pads tensors with 0s to match the largest tensor.
    :param tensors: The tensors.
    :return: The stacked tensor.
    """
    max_size = [0] * tensors[0].ndim
    for tensor in tensors:
        for i, shape in enumerate(tensor.shape):
            if shape > max_size[i]:
                max_size[i] = shape

    padded_tensors = []
    # Append 0s to each tensor
    for tensor in tensors:
        if all([d1 == d2 for d1, d2 in zip(tensor.shape, max_size)]):
            padded_tensors.append(tensor)
            continue
        # Calculate the padding for torch.nn.functional.pad
        padding = []
        for i in range(tensor.ndim):
            padding = [0, max_size[i] - tensor.shape[i]] + padding
        padded_tensors.append(
            torch.nn.functional.pad(tensor, padding, mode="constant", value=0)
        )
    return torch.stack(padded_tensors, dim=dim)


def resize_images(
    tensors: list[torch.Tensor],
    size: tuple[int, int],
    size_dim: int = 1,
    overlap_prop: float = 0,
) -> list[torch.Tensor]:
    """
    For each multidimensional image tensor, cut it into patches of the given size, optionally with overlap.
    :param tensors: The tensors.
    :param size: The desired dimensions.
    :param size_dim: The index to start searching for the size of the image.
    :return: The resized images.
    """
    output_tensors = []
    for tensor in tensors:
        curr_image_size = tensor.shape[size_dim : size_dim + 2]
        if all(
            [d1 <= d2 for d1, d2 in zip(curr_image_size, size)]
        ):  # Don't resize smaller/or equal images
            output_tensors.append(tensor)
            continue

        # Calculate the width/height starting indices for each patch
        for x_start in range(0, curr_image_size[0], int(size[0] * (1 - overlap_prop))):
            for y_start in range(
                0, curr_image_size[1], int(size[1] * (1 - overlap_prop))
            ):
                x_end = min(x_start + size[0], curr_image_size[0])
                y_end = min(y_start + size[1], curr_image_size[1])
                selector = [slice(None)] * size_dim + [
                    slice(x_start, x_end),
                    slice(y_start, y_end),
                ]
                output_tensors.append(tensor.__getitem__(tuple(selector)))
    return output_tensors


# /work/tansey/sanayeia/IMC_Data/stacks
def TiffDataset(
    directory: str = "/work/tansey/sanayeia/IMC_Data/stacks",
    size: Optional[tuple[int, int]] = None,
    overlap_prop: float = 0,
    include_panorama: bool = False,
) -> utils.data.Dataset:
    """
    Build a dataset using a directory of OME-TIFF files, where it is assumed that channels have the same semantic meaning.
    :return: The built dataset.

    Requires the tifffile and xml2dict packages.
    """
    import tifffile
    import xmltodict

    if include_panorama:
        print(
            "Warning: True panorama channels are not supported for OME-TIFFs, so it will be estimated as the mean of the other channels."
        )

    tensors = []
    channel_labels = []
    # for file in Path(directory).glob("*.tiff"):  - Doesn't follow symlinks
    for file in glob(
        os.path.join(directory, "*.tiff"),
        recursive=True,
    ):
        # Read the metadata and the data
        with tifffile.TiffFile(file) as tiff:
            metadata = tiff.ome_metadata
            metadata = xmltodict.parse(metadata)["OME"]["Image"]
            data = torch.tensor(tiff.asarray())

        channel_labels, tensors = normalize_channel_ordering(
            channel_labels,
            tensors,
            [channel["@Name"] for channel in metadata["Pixels"]["Channel"]],
            data,
        )
        continue

    channel_labels, tensors = clear_empty_channels(channel_labels, tensors)
    if len(channel_labels) == 0:
        return None

    if include_panorama:
        # Add a panorama channel to the beginning:
        channel_labels = ["panorama"] + channel_labels
        # For each image, insert a new dimension for the panorama
        for i, image in enumerate(tensors):
            tensors[i] = torch.cat(
                [torch.mean(image, dim=0).unsqueeze(0), image], dim=0
            )

    if size is not None:
        tensors = resize_images(tensors, size, overlap_prop=overlap_prop)
    dataset = utils.data.TensorDataset(safe_stack(tensors))
    dataset.channel_labels = channel_labels
    dataset.data = property(lambda self: self.tensors)
    dataset.reference_panorama_channel = 0 if include_panorama else None
    return dataset


# /work/tansey/pan_spatial/data/lung
def McdDataset(
    directory: str = "/work/tansey/pan_spatial/data/lung/imc",
    size: Optional[tuple[int, int]] = None,
    overlap_prop: float = 0,
    include_panorama: bool = False,
) -> utils.data.Dataset:
    """
    Build a dataset using a directory of compiled .mcd files with associated metadata. These should have consistent
    probes.
    :return: The built dataset.

    Requires the imc_tools package.
    """
    from readimc import MCDFile

    total_channels = []
    total_data = []
    # for file in Path(directory).glob("*.mcd"): - Doesn't follow symlinks
    panoramas = []
    for file in glob(
        os.path.join(directory, "*.mcd"),
        recursive=True,
    ):
        with MCDFile(file) as mcd:
            for slide_idx, slide in enumerate(mcd.slides):
                for acquisition_idx, acquisition in enumerate(slide.acquisitions):
                    if acquisition_idx != acquisition_idx and slide_idx != slide_idx:
                        continue  # Is this needed? Taken from https://github.com/tansey-lab/imc-tools/blob/e69b757c80b171d73b9c4dbaa57e61ac78f9ad45/src/imc_tools/images.py#L110
                    acquisition_data = mcd.read_acquisition(acquisition)
                    channel_names = [
                        label if label else name
                        for (label, name) in zip(
                            acquisition.channel_labels, acquisition.channel_names
                        )
                    ]
                    total_channels, total_data = normalize_channel_ordering(
                        total_channels,
                        total_data,
                        channel_names,
                        torch.tensor(acquisition_data),
                    )

                    if not include_panorama:
                        continue

                    if not acquisition.panorama:
                        panoramas.append(
                            torch.zeros(
                                acquisition_data.shape[1:],
                            )
                        )
                    else:
                        # Map a grey scaled panorama to the original channels
                        panorama_data = mcd.read_panorama(acquisition.panorama)
                        # Heavily based on https://github.com/tansey-lab/imc-tools/blob/main/src/imc_tools/mcd.py#L159
                        max_x = max(
                            [
                                float(acquisition.panorama.metadata["SlideX1PosUm"]),
                                float(acquisition.panorama.metadata["SlideX2PosUm"]),
                                float(acquisition.panorama.metadata["SlideX3PosUm"]),
                                float(acquisition.panorama.metadata["SlideX4PosUm"]),
                            ]
                        )

                        min_x = min(
                            [
                                float(acquisition.panorama.metadata["SlideX1PosUm"]),
                                float(acquisition.panorama.metadata["SlideX2PosUm"]),
                                float(acquisition.panorama.metadata["SlideX3PosUm"]),
                                float(acquisition.panorama.metadata["SlideX4PosUm"]),
                            ]
                        )

                        max_y = max(
                            [
                                float(acquisition.panorama.metadata["SlideY1PosUm"]),
                                float(acquisition.panorama.metadata["SlideY2PosUm"]),
                                float(acquisition.panorama.metadata["SlideY3PosUm"]),
                                float(acquisition.panorama.metadata["SlideY4PosUm"]),
                            ]
                        )

                        min_y = min(
                            [
                                float(acquisition.panorama.metadata["SlideY1PosUm"]),
                                float(acquisition.panorama.metadata["SlideY2PosUm"]),
                                float(acquisition.panorama.metadata["SlideY3PosUm"]),
                                float(acquisition.panorama.metadata["SlideY4PosUm"]),
                            ]
                        )

                        # convert x pixels to um
                        x_um_per_pixel = panorama_data.shape[0] / (max_x - min_x)
                        y_um_per_pixel = panorama_data.shape[1] / (max_y - min_y)

                        # Source: https://software.docs.hubmapconsortium.org/assays/imc.html
                        # "ROIStartXPosUm" and "ROIStartYPosUm"	Start X and Y-coordinates of the region of interest (µm).
                        # Note: This value must be divided by 1000 to correct for a bug (missing decimal point) in the Fluidigm software.
                        x1 = float(acquisition.metadata["ROIStartXPosUm"]) / 1000.0
                        y1 = float(acquisition.metadata["ROIStartYPosUm"]) / 1000.0
                        x2 = float(acquisition.metadata["ROIEndXPosUm"])
                        y2 = float(acquisition.metadata["ROIEndYPosUm"])

                        x_min_acq = min(x1, x2)
                        x_max_acq = max(x1, x2)
                        y_min_acq = min(y1, y2)
                        y_max_acq = max(y1, y2)

                        # Panorama is bigger than slide so we must crop and resize to match the acquisition
                        x_min_pan = int((x_min_acq - min_x) * x_um_per_pixel)
                        x_max_pan = int((x_max_acq - min_x) * x_um_per_pixel)
                        y_min_pan = int((y_min_acq - min_y) * y_um_per_pixel)
                        y_max_pan = int((y_max_acq - min_y) * y_um_per_pixel)

                        panorama_data = panorama_data[
                            x_min_pan:x_max_pan, y_min_pan:y_max_pan
                        ]
                        panorama_data = torch.tensor(panorama_data)
                        panorama_data = torchvision.transforms.v2.functional.resize(
                            panorama_data, acquisition_data.shape[1:]
                        )
                        # Collapse into a greyscale image
                        panorama_data = torch.mean(panorama_data, dim=0)
                        panoramas.append(panorama_data)

    total_channels, total_data = clear_empty_channels(total_channels, total_data)
    if len(total_channels) == 0:
        return None
    # Add panorama channels to the beginning:
    if include_panorama:
        total_channels = ["panorama"] + total_channels
        # For each image, insert a new dimension for the panorama
        for i, image in enumerate(total_data):
            total_data[i] = torch.cat([panoramas[i].unsqueeze(0), image], dim=0)
    if size is not None:
        total_data = resize_images(total_data, size, overlap_prop=overlap_prop)
    dataset = utils.data.TensorDataset(safe_stack(total_data))
    dataset.channel_labels = total_channels
    if include_panorama:
        dataset.reference_panorama_channel = 0
    else:
        dataset.reference_panorama_channel = None
    return dataset


def fuse_imc_datasets(
    *datasets: Union[TiffDataset, McdDataset], intersect_channels: bool = False
) -> utils.data.Dataset:
    """
    Fuse multiple IMC datasets together. This will attempt to match channel mismatches as much as possible.
    :param datasets: The datasets to fuse.
    :param intersect_channels: If True, the only remaining channels are the ones that are common to all datasets.
        Otherwise, the final dataset contains all channels, with missing channels filled to 0s (the default).
    :return: The fused dataset.
    """
    all_channels = []
    for dataset in datasets:
        all_channels.append(dataset.channel_labels)
    if intersect_channels:
        common_channels = set(all_channels[0])
        for channels in all_channels[1:]:
            common_channels = common_channels.intersection(channels)
        all_channels = list(common_channels)
    else:
        all_channels = list(set(itertools.chain(*all_channels)))

    all_data = []
    for dataset in datasets:
        # If channels and ordering exactly match, don't need to do anything
        if all(
            [
                len(dataset.channel_labels) > i and channel == dataset.channel_labels[i]
                for i, channel in enumerate(all_channels)
            ]
        ):
            all_data.append(dataset.tensors[0])
            continue
        # Otherwise, we need to re-order and potentially backfill with 0s
        dataset_shape = dataset.tensors[0].shape
        updated_shape = (dataset_shape[0], len(all_channels), *dataset_shape[2:])
        fixed_data = torch.zeros(updated_shape)
        for i, channel in enumerate(all_channels):
            if channel not in dataset.channel_labels:
                continue  # Already 0
            else:
                fixed_data[:, i, :, :] = dataset.tensors[0][
                    :, dataset.channel_labels.index(channel), :, :
                ]
        all_data.append(fixed_data)

    dataset = utils.data.TensorDataset(torch.cat(all_data, dim=0))
    dataset.channel_labels = all_channels
    return dataset


def CompressedImcDataset(
    compressed: torch.Tensor,
    uncompressed: torch.Tensor,
    assignments: torch.Tensor,
    uncompressed_labels: torch.Tensor,
    mask: torch.Tensor,
) -> utils.data.Dataset:
    dataset = utils.data.TensorDataset(compressed, uncompressed, mask)
    dataset.assignments = assignments
    dataset.uncompressed_labels = uncompressed_labels
    return dataset


def synthetic_compression(
    data: torch.tensor, n_compressed_channels: int = None
) -> tuple[torch.tensor, torch.tensor]:
    """
    Synthetically compress the signals to be pairwise combinations of the original signals.
    :param data: Original data (n_channels x n_pixels x n_pixels). Each channel is an observation.
    :param n_compressed_channels: The number of compressed channels to use. If None, this will be determined automatically.
    :return: Compressed data (n_compressed_channels x n_pixels x n_pixels), and the observation to compressed channel
        assignments. Each channel can contain multiple observations and each observation is replicated to exactly 2
        channels.
    """
    # First, calculate the number of required compressed channels if we want to use 2 signals per observation.
    n_obs = data.shape[0]
    n_channels = data.shape[1]
    n_compressed_channels = (
        torch.ceil(torch.log2(torch.tensor(n_channels))).int()
        if n_compressed_channels is None
        else n_compressed_channels
    )

    # Get the possible pairwise combinations without replacement
    compressed_assignments = torch.zeros(
        (n_channels, n_compressed_channels), dtype=bool
    )
    compressed_data = torch.zeros((n_obs, n_compressed_channels, *data.shape[2:]))
    curr_compressed_channel = 0
    for channel in itertools.chain(*itertools.repeat(range(n_channels), 2)):
        compressed_assignments[channel, curr_compressed_channel] = True
        compressed_data[:, curr_compressed_channel, :, :] += data[:, channel, :, :]
        curr_compressed_channel += 1
        if curr_compressed_channel == n_compressed_channels:
            curr_compressed_channel = 0

    return compressed_data, compressed_assignments


def pad_imc_dataset(
    dataset: CompressedImcDataset, padding: int
) -> CompressedImcDataset:
    data = dataset.tensors[0]
    # Pad the last two dimensions (pixels) with 0s
    padding_before = padding // 2
    padding_after = padding - padding_before
    # Note: The mask should end up getting padded with zeros, indicating that the pixels are not valid
    data = torch.nn.functional.pad(
        data,
        (padding_before, padding_after, padding_before, padding_after),
        mode="constant",
        value=0,
    )
    dataset.tensors = (data,)
    return dataset


def _compress_dataset(
    dataset: utils.data.TensorDataset, codebook: torch.Tensor
) -> CompressedImcDataset:
    uncompressed = dataset.tensors[0]
    if codebook is None:
        compressed, codebook = synthetic_compression(uncompressed)
    else:
        compressed = torch.matmul(uncompressed, codebook)

    return (
        CompressedImcDataset(
            compressed,
            uncompressed,
            codebook,
            dataset.channel_labels,
            compressed.new_ones((compressed.shape[0], *compressed.shape[2:])),
        ),
        codebook,
    )


def _generate_dataset(
    tiffs: list[str],
    mcds: list[str],
    image_size: Optional[tuple[int, int]],
    patch_overlap: float,
    codebook: torch.Tensor,
    out_dir: str,
):
    """
    Harmonizes and saves the data into a static directory that can be used later.
    :param tiffs: The directory to search for ome.tiff files.
    :param mcds: The directory to search for .mcd files.
    :param image_size: The size of the images to use.
    :param patch_overlap: The amount of overlap between patches when resized.
    :param codebook: The codebook, if known.
    :param out_dir: The output directory.
    """
    out_dir = Path(out_dir)

    datasets = []
    for tiff in tiffs:
        dataset = TiffDataset(tiff, image_size, patch_overlap)
        if dataset:
            datasets.append(dataset)
    for mcd in mcds:
        dataset = McdDataset(mcd, image_size, patch_overlap)
        if dataset:
            datasets.append(dataset)
    assert len(datasets) > 0, f"No datasets found in {tiffs} or {mcds}"
    fused = fuse_imc_datasets(*datasets)
    size = tuple(fused.tensors[0].shape[2:])
    # Must be square
    assert size[0] == size[1]
    # The patching process will break if the size is not divisible by 32, so we must pad with 0s  FIXME: Adjust stride to reduce the need for this
    if size[0] % 32 != 0:
        fused = pad_imc_dataset(fused, 32 - (size[0] % 32))
        size = tuple(fused.tensors[0].shape[2:])
    n_proteins = len(fused.channel_labels)
    # Compression
    compressed, codebook = _compress_dataset(fused, codebook)
    n_channels = compressed.tensors[0].shape[1]
    codebook_labels = compressed.uncompressed_labels

    metadata = {
        "size": size,
        "n_channels": n_channels,
        "n_proteins": n_proteins,
        "codebook_labels": codebook_labels,
        "codebook": codebook,
    }

    out_dir.mkdir(exist_ok=True)
    torch.save(metadata, out_dir / "metadata.pt")
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for i, (compressed, uncompressed, mask) in enumerate(
        zip(compressed.tensors[0], compressed.tensors[1], compressed.tensors[2])
    ):
        image_dir = images_dir / f"{i}"
        image_dir.mkdir(exist_ok=True)
        torch.save(compressed.clone(), image_dir / "image_compressed.pt")
        torch.save(uncompressed.clone(), image_dir / "image_uncompressed.pt")
        torch.save(mask.clone(), image_dir / "mask.pt")


class DirectoryDataset(utils.data.Dataset):
    """
    Dataset that loads saved tensors from a directory, where subdirectories are each a datapoint.
    """

    def __init__(self, parent_dir: str, *tensor_patterns: str):
        self.parent_dir = Path(parent_dir)
        metadata = torch.load(self.parent_dir / "metadata.pt")
        self.size = metadata["size"]
        self.n_channels = metadata["n_channels"]
        self.n_proteins = metadata["n_proteins"]
        self.codebook_labels = metadata["codebook_labels"]
        self.codebook = metadata["codebook"]
        self.data_paths = []
        for f in self.parent_dir.glob("images/*"):
            if not f.is_dir():
                continue
            tensor_paths = []
            for pattern in tensor_patterns:
                tensor_paths.extend(list(f.glob(pattern)))
            if len(tensor_paths) < len(tensor_patterns):
                print(f"Skipping {f} because it does not contain all tensors.")
                continue
            self.data_paths.append(tensor_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        paths = self.data_paths[idx]
        return tuple([torch.load(path) for path in paths])


class SpikeInChannelTransform(nn.Module):
    """
    Augment the data by adding in a few channels of known uncompressed values.
    """

    def __init__(self, spike_in_channels: list[int]):
        super().__init__()
        self.spike_in_channels = spike_in_channels

    def forward(self, x):
        input, output, mask = x
        spike_in = output[self.spike_in_channels, :, :]
        input = torch.cat([input, spike_in], dim=0)
        return input, output, mask


class LambdaDataset(utils.data.Dataset):
    """
    Apply a function to a dataset.
    """

    def __init__(
        self, dataset: utils.data.Dataset, total_func, input_func, prediction_func
    ):
        self.dataset = dataset
        self.total_func = total_func
        self.input_func = input_func
        self.prediction_func = prediction_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        initial_input, initial_output, initial_mask = self.total_func(self.dataset[idx])
        if self.input_func is not None:
            initial_input = self.input_func(initial_input)
        if self.prediction_func is not None:
            initial_output = self.prediction_func(initial_output)
        return initial_input, initial_output, initial_mask


class InSilicoCompressedImcDataset(pl.LightningDataModule):
    def __init__(
        self,
        tiffs: list[str] = ("/work/tansey/sanayeia/IMC_Data/",),
        mcds: list[str] = ("/work/tansey/pan_spatial/data/lung",),
        image_size: Optional[tuple[int, int]] = None,
        patch_overlap: float = 0,
        save_dir: str = None,
        batch_size: int = 2,
        seed: int = 12345,
        codebook: torch.Tensor = None,
        generate: bool = True,
        random_flip: float = 0,
        random_rotate: float = 0,
        random_erase: float = 0,
        random_shear: float = 0,
        random_translate: float = 0,
        random_scale: float = 0,
        spike_in_channels: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.tiffs = tiffs
        self.mcds = mcds
        self.size = None
        self.n_channels = None
        self.n_proteins = None
        self.codebook = codebook
        self.codebook_labels = None
        self.initialized = False
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_erase = random_erase
        self.random_shear = random_shear
        self.random_translate = random_translate
        self.random_scale = random_scale
        self.spike_in_channels = spike_in_channels

        if save_dir is None:
            all_dirs = [os.path.basename(dir) for dir in tiffs + mcds]
            hashed = hash(tuple(all_dirs))
            self.save_dir = f"data_{hashed}"
        else:
            self.save_dir = save_dir

        if generate and not os.path.exists(self.save_dir):
            _generate_dataset(
                self.tiffs,
                self.mcds,
                image_size,  # Only used when generating the dataset
                patch_overlap,  # Only used when generating the dataset
                self.codebook,
                self.save_dir,
            )

    def prepare_data(self):
        if self.initialized:
            return

        dataset = DirectoryDataset(
            self.save_dir, "image_compressed.pt", "image_uncompressed.pt", "mask.pt"
        )

        self.size = dataset.size
        self.n_channels = dataset.n_channels
        self.n_proteins = dataset.n_proteins
        self.codebook_labels = dataset.codebook_labels
        self.codebook = dataset.codebook

        if self.spike_in_channels > 0:
            self.spike_in_channels = torch.randperm(self.n_proteins)[
                : self.spike_in_channels
            ]
            dataset = LambdaDataset(
                dataset,
                SpikeInChannelTransform(self.spike_in_channels),
                None,
                None,
            )
        else:
            self.spike_in_channels = []

        self.train, self.val, self.test = utils.data.random_split(
            dataset,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if (
            self.random_flip > 0
            or self.random_rotate > 0
            or self.random_erase > 0
            or self.random_shear > 0
            or self.random_translate > 0
            or self.random_scale > 0
        ):
            transforms = []
            if self.random_flip > 0:
                transforms.append(RandomHorizontalFlip(p=self.random_flip))
                transforms.append(RandomVerticalFlip(p=self.random_flip))

            degrees = self.random_rotate
            translate = (
                None
                if self.random_translate == 0
                else (self.random_translate, self.random_translate)
            )
            scale = (
                None
                if self.random_scale == 0
                else (1 - self.random_scale, 1 + self.random_scale)
            )
            shear = None if self.random_shear == 0 else self.random_shear

            transforms.append(
                RandomAffine(
                    degrees=degrees, translate=translate, scale=scale, shear=shear
                )
            )
            transforms = Compose(transforms)
            self.train = LambdaDataset(
                self.train,
                transforms,
                RandomErasing(p=self.random_erase)
                if self.random_erase > 0
                else None,  # Only apply erasing to the input
                None,
            )

        self.initialized = True

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return utils.data.DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False
        )

    def teardown(self, stage: str) -> None:
        pass


if __name__ == "__main__":
    import docopt

    args = docopt.docopt(__doc__)

    image_size = args["--image_size"]
    if image_size is None or image_size == "None":
        image_size = None
    else:
        image_size = (int(image_size), int(image_size))
    patch_overlap = float(args["--patch_overlap"])

    dataset_path = args["<dataset_path>"]
    output_path = args["<output_path>"]
    data = InSilicoCompressedImcDataset(
        tiffs=[dataset_path],
        mcds=[dataset_path],
        save_dir=output_path,
        # TODO: Test sensitivity
        image_size=image_size,
        patch_overlap=patch_overlap,
        generate=True,
    )
    # Initializing the dataset will generate the data
