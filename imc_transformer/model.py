import os.path

from hiera import MaskedAutoencoderHiera
from timm.scheduler import CosineLRScheduler

from data import InSilicoCompressedImcDataset

import lightning.pytorch as pl
from torch import nn, optim, utils, Tensor
import torch


class ImcTransformer(pl.LightningModule):
    """
    Model definition for the IMC Transformer.
    The transformer uses a base architecture inspired by Hiera: https://github.com/facebookresearch/hiera
    With some additional channel handling inspired by ChannelViT: https://github.com/insitro/ChannelViT

    Following the Hiera paper, we will do a two part training process:
    1. Pre-Train the model as a masked autoencoder (Most of the training).
    2. Exchange the head to a regression head and fine-tune the model on the downstream task.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        n_channels: int,
        n_proteins: int,
        deepspeed: bool = False,
    ):
        """
        Initialize the model.

        :param input_size: Size of each input image (n_pixels x n_pixels).
        :param n_channels: The number of channels in the input data.
        :param n_proteins: The number of proteins to decode (i.e. output channel count).
        :param deepspeed: Whether to use deepspeed stage 2 efficiency optimizations within the model.
        """
        super().__init__()
        self.optim_args = {"lr": 8e-4, "weight_decay": 0.05, "betas": (0.9, 0.95)}
        self.masked_training_step = True
        self.deepspeed = deepspeed
        self.hiera = MaskedAutoencoderHiera(
            out_chans=n_proteins,
            in_chans=n_channels,
            input_size=input_size,
        )

    def convert_to_regression(self):
        self.masked_training_step = False
        self.hiera.enable_downstream_regression()
        self.optim_args["lr"] = 2e-3
        self.optim_args["betas"] = (0.9, 0.999)

    def training_step(self, batch, batch_idx):
        return self.step_impl("training", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step_impl("validation", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step_impl("test", batch, batch_idx)

    def step_impl(self, kind: str, batch, batch_idx):
        compressed, uncompressed = batch

        if self.masked_training_step:
            loss, pred, label, mask = self.hiera(compressed)
            self.log(
                f"{kind}_MAE_loss", loss, prog_bar=True, on_step=True, on_epoch=True
            )
        else:
            # Regression loss
            pred, mask = self.hiera(compressed)
            loss = nn.functional.mse_loss(pred, uncompressed)
            self.log(
                f"{kind}_PRED_loss", loss, prog_bar=True, on_step=True, on_epoch=True
            )

        return loss

    def configure_optimizers(self):
        # AdamW with Cosine LR Decay
        if self.deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), **self.optim_args)
        else:
            optimizer = optim.AdamW(self.parameters(), **self.optim_args)
        scheduler = CosineLRScheduler(
            optimizer, t_initial=10, warmup_t=40 if self.masked_training_step else 5
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric):
        scheduler.step(epoch=self.current_epoch, metric=metric)


def train_model(deepspeed: bool = True):
    """
    Train the model.
    :param deepspeed: Whether to use deepspeed efficiency optimizations.
    """
    if os.path.exists("/data"):
        dataset = InSilicoCompressedImcDataset(tiffs=["/data"], mcds=["/data"])
    else:
        dataset = InSilicoCompressedImcDataset(tiffs=["./raw"], mcds=["./raw"])
    dataset.prepare_data()

    model = ImcTransformer(
        input_size=dataset.size,
        n_channels=dataset.n_channels,
        n_proteins=dataset.n_proteins,
        deepspeed=deepspeed,
    )
    core_trainer_kwargs = dict(devices=1, accelerator="cpu")
    if torch.cuda.is_available():
        core_trainer_kwargs["accelerator"] = "gpu"

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

    mae_trainer = pl.Trainer(
        **core_trainer_kwargs,
        detect_anomaly=True,
        num_sanity_val_steps=0,
        max_epochs=800,
        # Enable checkpointing:
        # checkpoint_callback=pl.callbacks.ModelCheckpoint(
        #     monitor="validation_MAE_loss",
        #     mode="min",
        #     save_top_k=1,
        #     save_last=True,
        #     filename="mae-{epoch:02d}-{validation_MAE_loss:.2f}",
        # ),
    )
    reg_trainer = pl.Trainer(
        **core_trainer_kwargs,
        detect_anomaly=True,
        num_sanity_val_steps=0,
        max_epochs=150,
    )

    print("MAE Pre-Training")
    mae_trainer.fit(model, dataset)
    print("Regression Fine-Tuning")
    model.convert_to_regression()
    reg_trainer.fit(model, dataset)

    model.eval()
    mae_trainer.test(model, dataset)
    return model


def main():
    train_model()


if __name__ == "__main__":
    train_model()
