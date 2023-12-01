"""Train the IMC Transformer model.

Usage:
    model.py <data_dir> <output_dir> [--epochs=<n>] [--deepspeed] [--random_flip_augmentation=<p>] [--random_rotate_augmentation=<deg>] [--random_erasing=<p>] [--random_shear_augmentation=<deg>] [--random_translate_augmentation=<frac>] [--random_scale_augmentation=<frac>] [--spike_in_channels=<spike_in_channels>]
    model.py (-h | --help)

Options:
    -h --help       Show this screen.
    --epochs=<n>    Number of epochs to train for [default: 100].
    --deepspeed     Use deepspeed optimizations.
    --random_flip_augmentation=<p>   Probability of random flip augmentation during training [default: 0].
    --random_rotate_augmentation=<p>   Degrees of random rotation augmentation during training [default: 0].
    --random_erasing=<p>   Probability of random erasing augmentation during training [default: 0].
    --random_shear_augmentation=<deg>   Degrees of random shear augmentation during training [default: 0].
    --random_translate_augmentation=<frac>   Fraction of random translation augmentation during training [default: 0].
    --random_scale_augmentation=<frac>   Fraction of random scale augmentation during training (1 - frac) to (1 + frac)  [default: 0].
    --spike_in_channels=<spike_in_channels>   Number of channels to spike in [default: 0].
"""
import sys
from pathlib import Path
from typing import Union

import numpy as np
import timm
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from data import InSilicoCompressedImcDataset

import lightning.pytorch as pl
from torch import nn, optim, utils, Tensor
import torch

from diffusion import DiffusionIMC, DiffusionUnet
from unet import UNet
from convnext import ConvNeXtUnet
from vanilla import CollatedVanillaCNN


class ImcNet(pl.LightningModule):
    """
    Model definition for the IMC Net.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        n_channels: int,
        n_proteins: int,
        deepspeed: bool = False,
        model: str = "diffusion",
        sanity_check: bool = False,
    ):
        """
        Initialize the model.

        :param input_size: Size of each input image (n_pixels x n_pixels).
        :param n_channels: The number of channels in the input data.
        :param n_proteins: The number of proteins to decode (i.e. output channel count).
        :param deepspeed: Whether to use deepspeed stage 2 efficiency optimizations within the model.
        """
        super().__init__()
        self.optim_args = {"lr": 2e-4}
        self.deepspeed = deepspeed
        self.sanity_check = sanity_check
        self.model = model
        if sanity_check:
            n_proteins = n_proteins // 4
            n_channels = n_proteins
            # Random channel reordering
            self.channel_reordering = torch.randperm(n_channels)

        if model == "unet":
            print("Using UNET")
            self.module = UNet(
                in_chans=n_channels,
                out_chans=n_proteins,
                image_size=input_size,
                attention=False,
            )
        elif model == "convnext":
            print("Using ConvNeXt")
            self.module = ConvNeXtUnet(
                channels=n_channels, out_channels=n_proteins, dropout=0.1
            )
        elif model == "diffusion":
            print("Using Diffusion")
            self.module = DiffusionIMC(
                compressed_channels=n_channels,
                uncompressed_channels=n_proteins,
                n_steps=1_000,
                # sampler='ddim',  # Seems broken?  Refer to this implementation https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
                sampler="jump",
                # sampler="ddpm",  # FIXME: Poisson loss for ddpm
                raw_counts=True,
                unet=DiffusionUnet(
                    image_channels=n_proteins,
                    conditional_channels=n_channels,
                    out_image_channels=n_proteins,
                    feature_channels=128,
                    feature_mult=(1, 1, 1),
                    use_attn=(False, True, True),
                    n_blocks=3,
                    n_groups=32,
                    dropout=0,
                    activation=nn.SiLU,
                    n_heads=1,
                    embedding_type="shift_and_scale",
                ),
            )
        else:
            print("Using Vanilla CNN")
            self.module = CollatedVanillaCNN(
                in_channels=n_channels, out_channels=n_proteins
            )

    def training_step(self, batch, batch_idx):
        return self.step_impl("training", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step_impl("validation", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step_impl("test", batch, batch_idx)

    def forward(self, batch, intermediate=False):
        compressed, uncompressed, mask = batch
        if self.sanity_check:
            # Only use half the channels for computational efficiency
            uncompressed = uncompressed[:, 0 : self.channel_reordering.shape[0]]
            compressed = uncompressed[:, self.channel_reordering]
        if self.model == "diffusion":
            # Normalize to log space to make it easier for the model
            return self.module.sample_image(
                compressed.to(self.module.device), intermediate
            )
        else:
            return self.module(compressed)

    def step_impl(self, kind: str, batch, batch_idx):
        compressed, uncompressed, mask = batch

        if self.sanity_check:
            # Only use half the channels for computational efficiency
            uncompressed = uncompressed[:, 0 : self.channel_reordering.shape[0]]
            compressed = uncompressed[:, self.channel_reordering]

        # TODO: Should there be an approximate loss term added
        # Where we try to recreate the compressed image from our predicted
        # true images?
        if self.model == "diffusion":
            loss = self.module.loss(uncompressed, compressed, mask)
        else:
            pred = self.module(compressed)

            # loss = nn.functional.mse_loss(pred, uncompressed)
            loss = nn.functional.l1_loss(pred, uncompressed, reduction="none")
            # We have counts data, so use Poisson loss
            # loss = nn.functional.poisson_nll_loss(pred, uncompressed, log_input=False)

            # Regularization term that encourages sparsity in output
            # regularization = (
            #     torch.mean((pred * (uncompressed < 1)) + ((pred < 1) * (uncompressed > 0)))
            #     + 1
            # )
            # miss_classified_zeros_selector = (pred >= 1) * (uncompressed < 1)
            # regularization = (miss_classified_zeros_selector * (pred * pred)).sum() / (
            #     miss_classified_zeros_selector.sum() + 1
            # )  # Squared penalty by how far off we are
            # if kind == "training":
            #     self.log(
            #         f"{kind}_reg",
            #         regularization,
            #         prog_bar=True,
            #         on_step=True,
            #         on_epoch=False,
            #     )
            #
            # loss = (
            #     loss * regularization
            # )  # torch.maximum(regularization, torch.ones_like(regularization))

            # Handle masking
            loss = (loss * mask.unsqueeze(1).float()).sum() / mask.sum()

        self.log(f"{kind}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Adam with Cosine LR Decay
        if self.deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), **self.optim_args)
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                betas=(0.9, 0.999),
                weight_decay=1e-3,
                **self.optim_args,
            )
            return optimizer  #  TODO : Add scheduler
            # optimizer = optim.SGD(self.parameters(), **self.optim_args, momentum=0.9)
        scheduler = CosineLRScheduler(optimizer, t_initial=3, warmup_t=2)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric):
        scheduler.step(epoch=self.current_epoch, metric=metric)


def train_model(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    deepspeed: bool = False,
    random_flip_augmentation: float = 0,
    random_rotate_augmentation: float = 0,
    random_erasing_augmentation: float = 0,
    random_shear_augmentation: float = 0,
    random_translate_augmentation: float = 0,
    random_scale_augmentation: float = 0,
    spike_in_channels: int = 0,
    sanity_check: bool = False,
):
    """
    Train the model.
    :param deepspeed: Whether to use deepspeed efficiency optimizations.
    """
    dataset = InSilicoCompressedImcDataset(
        batch_size=batch_size,
        tiffs=[],
        mcds=[],
        save_dir=data_dir,
        generate=False,
        random_flip=random_flip_augmentation,
        random_rotate=random_rotate_augmentation,
        random_erase=random_erasing_augmentation,
        random_shear=random_shear_augmentation,
        random_translate=random_translate_augmentation,
        random_scale=random_scale_augmentation,
        spike_in_channels=spike_in_channels,
        # TODO: Random Affine transform?
    )
    dataset.prepare_data()

    # model = ImcTransformer(
    #     input_size=dataset.size,
    #     n_channels=dataset.n_channels,
    #     n_proteins=dataset.n_proteins,
    #     deepspeed=deepspeed,
    # )
    model = ImcNet(
        input_size=dataset.size,
        n_channels=dataset.n_channels + spike_in_channels,
        n_proteins=dataset.n_proteins,
        deepspeed=deepspeed,
        sanity_check=sanity_check,
    )
    core_trainer_kwargs = dict(devices=1)  # , accelerator="cpu")
    if torch.cuda.is_available():
        core_trainer_kwargs["accelerator"] = "gpu"
    # elif getattr(sys, "gettrace", None) is not None or sys.gettrace() is not None:
    #     # Force CPU if in debugger, otherwise MPS will crash
    #     core_trainer_kwargs["accelerator"] = "cpu"

    if deepspeed:
        from lightning.pytorch.strategies import DeepSpeedStrategy

        core_trainer_kwargs["strategy"] = DeepSpeedStrategy(
            accelerator=core_trainer_kwargs.get("accelerator", "cpu"),
            zero_optimization=True,
            stage=2,
            offload_optimizer=True,
            offload_parameters=True,
            pin_memory=True,
        )
        core_trainer_kwargs["accelerator"] = "auto"
        core_trainer_kwargs["precision"] = 16

    # mae_trainer = pl.Trainer(
    #     **core_trainer_kwargs,
    #     # detect_anomaly=True,
    # num_sanity_val_steps=0,
    #     max_epochs=800,
    #     callbacks=[
    #         ModelCheckpoint(
    #             dirpath="mae_pretrain/",
    #             save_last=True,
    #             every_n_epochs=5,
    #             enable_version_counter=False,
    #         )
    #     ],
    # )
    # TODO: EMA 0.999
    reg_trainer = pl.Trainer(
        **core_trainer_kwargs,
        detect_anomaly=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1,
        max_epochs=epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath="reg_finetune/",
                save_last=True,
                every_n_epochs=2,
                enable_version_counter=True,
            )
        ],
    )

    # print("MAE Pre-Training")
    # mae_trainer.fit(model, dataset, ckpt_path="last")
    # print("Regression Fine-Tuning")
    # model.convert_to_regression()
    reg_trainer.fit(model, dataset, ckpt_path="last")

    model.eval()
    # reg_trainer.test(model, dataset)
    return model, dataset


def visual_clip(x, p=0.95):
    """
    Clips the data for plotting. By default, clips the top 2% of the data.
    """
    clip_val = np.quantile(x, p)
    # Don't clip if the clip_val leads to an empty array
    if clip_val <= x.min():
        return x
    return np.clip(x, 0, clip_val)


def _make_pdf(
    original_data: torch.Tensor,
    compressed_data: torch.Tensor,
    predicted_data: Union[torch.Tensor, list[torch.Tensor]],
    codebook: torch.Tensor,
    codebook_labels: list[str],
    file: str,
    spike_in_channels: list[int] = [],
    sanity_check: bool = False,
):
    """
    Generates a basic PDF report for a given sample image where each page depicts the original image, the compressed
    images for the channels the input is associated with, and the reconstructed image.
    :param original_data: The original input.
    :param compressed_data: The compressed input.
    :param predicted_data: The predicted output.
    :param codebook: The mapping from input channels to compressed channels.
    :param codebook_labels: The names of the input channels.
    :param file: The file to write to.
    :param spike_in_channels: The number of channels to ignore.
    :param sanity_check: Whether this is a sanity check run, i.e. ignore the "compressed" data since its just reordered uncompressed.
    """
    from datetime import datetime

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    if sanity_check:
        spike_in_channels = list(model.channel_reordering.cpu().numpy())

    with PdfPages(file) as pdf:
        d = pdf.infodict()
        d["Title"] = "IMC Reconstruction Example"
        d["Author"] = "IMC NET"
        d["CreationDate"] = datetime.today()
        d["ModDate"] = datetime.today()

        in_channels = original_data.shape[0]
        out_channels = max(compressed_data.shape[0] - len(spike_in_channels), 2)
        for i, (in_channel, channel_name) in enumerate(
            zip(range(in_channels), codebook_labels)
        ):
            if sanity_check and in_channel not in spike_in_channels:
                continue

            # 3 Rows: Original image channel, the corresponding compressed images, and the final predicted image
            fig, axs = plt.subplots(
                nrows=3 if not sanity_check else 2,
                ncols=out_channels,
                figsize=(out_channels * 10, 30),
                dpi=300,
            )
            fig.suptitle(
                f"{channel_name} ({in_channel})",
                fontsize=32,
            )
            middle_index = out_channels // 2 - 1
            # Original image
            axis = axs[0, middle_index]
            title = "Original"
            if in_channel in spike_in_channels:
                title += " (Spike-In)"
            axis.set_title(
                title,
                fontsize=24,
                fontweight="bold" if in_channel in spike_in_channels else "normal",
            )
            clipped_original = visual_clip(original_data[in_channel].cpu().numpy())
            axis.imshow(
                clipped_original,
                cmap="magma",
                vmin=0,
                # norm=matplotlib.colors.LogNorm(),
            )
            axis.axis("off")
            if not sanity_check:
                # Make other images in the row blank
                for out_channel in range(out_channels):
                    if out_channel != middle_index:
                        axs[0, out_channel].axis("off")
                # Compressed images
                for out_channel in range(out_channels):
                    axis = axs[1, out_channel]
                    axis.set_title(
                        f"Compressed {out_channel}, Contains Input: {codebook[in_channel, out_channel].item()}",
                        fontsize=24,
                        fontweight="bold"
                        if codebook[in_channel, out_channel].item()
                        else "normal",
                    )
                    axis.imshow(
                        visual_clip(compressed_data[out_channel].cpu().numpy()),
                        cmap="magma",
                        vmin=0,
                        # norm=matplotlib.colors.LogNorm(),
                    )
                    axis.axis("off")
            # Predicted image
            if isinstance(predicted_data, list):
                for ti in range(out_channels):
                    axis = axs[2 if not sanity_check else 1, ti]
                    if ti < out_channels - 1:
                        # Convert to diffusion time step
                        ti = int((ti / out_channels) * len(predicted_data))
                    else:
                        ti = len(predicted_data) - 1
                        # Ensure final time step is included
                    axis.set_title(
                        f"Predicted T={ti}",
                        fontsize=24,
                    )
                    axis.imshow(
                        predicted_data[ti][in_channel].cpu().numpy(),
                        cmap="magma",
                        vmin=0,
                        # norm=matplotlib.colors.LogNorm(),
                    )
                    axis.axis("off")
            else:
                axis = axs[2 if not sanity_check else 1, middle_index]
                axis.set_title("Predicted", fontsize=24)
                axis.imshow(
                    predicted_data[in_channel].cpu().numpy(),
                    cmap="magma",
                    vmin=0,
                    # norm=matplotlib.colors.LogNorm(),
                )
                axis.axis("off")
                # Make other images in the row blank
                for out_channel in range(out_channels):
                    if out_channel != middle_index:
                        axs[2 if not sanity_check else 1, out_channel].axis("off")
            pdf.attach_note(
                f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            pdf.savefig()
            plt.close()


@torch.no_grad()
def generate_examples(
    model: ImcNet,
    dataset: InSilicoCompressedImcDataset,
    output_dir: str,
    n_samples: int = 10,
):
    """
    Pick a couple random examples from the testing dataset, and generate predictions.
    This outputs a set of PDFs showing the various input channels, the original channels and the predicted channels.
    :param model: The trained model.
    :param dataset: The dataset.
    :param output_dir: The output directory.
    :param n_samples: The number of samples to look at.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    model.eval()
    dataset.prepare_data()
    test_dataset = dataset.test
    # Shuffle the test dataset
    test_dataset = utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)))
    dataloader = utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    for i, (compressed, uncompressed, mask) in tqdm(
        enumerate(dataloader), total=n_samples, desc="Generating Examples"
    ):
        if i >= n_samples:
            break
        if model.model == "diffusion":
            pred = [
                p.squeeze(0)
                for p in model.forward(
                    (compressed, uncompressed, mask), intermediate=True
                )
            ]
        else:
            pred = model.forward((compressed, uncompressed, mask)).squeeze(0)
        _make_pdf(
            uncompressed.squeeze(0),
            uncompressed[:, model.channel_reordering].squeeze(0)
            if model.sanity_check
            else compressed.squeeze(0),
            pred,
            dataset.codebook,
            dataset.codebook_labels,
            output_dir / f"sample_{i}.pdf",
            dataset.spike_in_channels,
            sanity_check=model.sanity_check,
        )


if __name__ == "__main__":
    import docopt

    args = docopt.docopt(__doc__)

    model, dataset = train_model(
        data_dir=args["<data_dir>"],
        epochs=int(args["--epochs"]),
        deepspeed=args["--deepspeed"],
        random_flip_augmentation=float(args["--random_flip_augmentation"]),
        random_rotate_augmentation=float(args["--random_rotate_augmentation"]),
        random_erasing_augmentation=float(args["--random_erasing"]),
        random_shear_augmentation=float(args["--random_shear_augmentation"]),
        random_translate_augmentation=float(args["--random_translate_augmentation"]),
        random_scale_augmentation=float(args["--random_scale_augmentation"]),
        spike_in_channels=int(args["--spike_in_channels"]),
        sanity_check=False,
    )

    generate_examples(
        model=model,
        dataset=dataset,
        output_dir=args["<output_dir>"],
        n_samples=10,
    )
