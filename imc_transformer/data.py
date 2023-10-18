import itertools
from typing import Union
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn, optim, utils


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


# /work/tansey/sanayeia/IMC_Data/stacks
def TiffDataset(
    directory: str = "/work/tansey/sanayeia/IMC_Data/",
) -> utils.data.Dataset:
    """
    Build a dataset using a directory of OME-TIFF files, where it is assumed that channels have the same semantic meaning.
    :return: The built dataset.

    Requires the tifffile and xml2dict packages.
    """
    import tifffile
    import xmltodict

    tensors = []
    channel_labels = []
    for file in Path(directory).glob("*.tiff"):
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
    dataset = utils.data.TensorDataset(torch.stack(tensors))
    dataset.channel_labels = channel_labels
    dataset.data = property(lambda self: self.tensors)
    return dataset


# /work/tansey/pan_spatial/data/lung
def McdDataset(
    directory: str = "/work/tansey/pan_spatial/data/lung",
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
    for file in Path(directory).glob("*.mcd"):
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

    total_channels, total_data = clear_empty_channels(total_channels, total_data)
    if len(total_channels) == 0:
        return None
    dataset = utils.data.TensorDataset(torch.stack(total_data))
    dataset.channel_labels = total_channels
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
) -> utils.data.Dataset:
    dataset = utils.data.TensorDataset(compressed, uncompressed)
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
    data = torch.nn.functional.pad(
        data,
        (padding_before, padding_after, padding_before, padding_after),
        mode="constant",
        value=0,
    )
    dataset.tensors = (data,)
    return dataset


class InSilicoCompressedImcDataset(pl.LightningDataModule):
    def __init__(
        self,
        tiffs: list[str] = ("/work/tansey/sanayeia/IMC_Data/",),
        mcds: list[str] = ("/work/tansey/pan_spatial/data/lung",),
        batch_size: int = 32,
        seed: int = 12345,
        codebook: torch.Tensor = None,
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

    def prepare_data(self):
        if self.initialized:
            return

        datasets = []
        for tiff in self.tiffs:
            dataset = TiffDataset(tiff)
            if dataset:
                datasets.append(dataset)
        for mcd in self.mcds:
            dataset = McdDataset(mcd)
            if dataset:
                datasets.append(dataset)
        fused = fuse_imc_datasets(*datasets)
        self.size = tuple(fused.tensors[0].shape[2:])
        # Must be square
        assert self.size[0] == self.size[1]
        # The patching process will break if the size is not divisible by 32, so we must pad with 0s  FIXME: Adjust stride
        if self.size[0] % 32 != 0:
            fused = pad_imc_dataset(fused, 32 - (self.size[0] % 32))
            self.size = tuple(fused.tensors[0].shape[2:])
        self.n_proteins = len(fused.channel_labels)
        # Compression
        compressed = self._compress_dataset(fused)
        self.n_channels = compressed.tensors[0].shape[1]
        self.codebook_labels = compressed.uncompressed_labels
        self.train, self.val, self.test = utils.data.random_split(
            compressed,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.initialized = True

    def _compress_dataset(
        self, dataset: utils.data.TensorDataset
    ) -> CompressedImcDataset:
        uncompressed = dataset.tensors[0]
        if self.codebook is None:
            compressed, self.codebook = synthetic_compression(uncompressed)
        else:
            compressed = torch.matmul(uncompressed, self.codebook)

        return CompressedImcDataset(
            compressed, uncompressed, self.codebook, dataset.channel_labels
        )

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True
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
    data = InSilicoCompressedImcDataset(tiffs=["raw/"], mcds=["raw/"])
    data.prepare_data()
    data.setup()
    print(data)
