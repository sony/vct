import lightning as L
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.utils import rescaling_inv, adjust_channels
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch import seed_everything


class FIDCallback(L.Callback):

    def __init__(self, sample_shape, n_iters, n_dataset_samples, every_n_iterations=1):
        super().__init__()
        self.sample_shape = sample_shape
        self.every_n_iterations = every_n_iterations  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.n_iters = n_iters
        self.n_dataset_samples = n_dataset_samples
        with isolate_rng():
            self.fid = FrechetInceptionDistance(reset_real_features=False, normalize=True)
        self.best = torch.inf

    def on_train_start(self, trainer, pl_module):
        with isolate_rng():
            seed_everything(32, workers=True)
            with torch.no_grad():
                self.fid = self.fid.to(pl_module.device)
                for batch in trainer.datamodule.fid_dataloader():
                    if isinstance(batch, list):
                        data = batch[0].to(pl_module.device)
                    else:
                        data = batch.to(pl_module.device)
                    # if data.dtype != torch.float32:
                    #     data = data.to(torch.float32) / 255.
                    data = adjust_channels(data)
                    self.fid.update(data, real=True)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_iterations == 0:
            with isolate_rng():
                seed_everything(32, workers=True)
                with torch.no_grad():
                    pl_module.model.eval()
                    self.fid = self.fid.to(pl_module.device)
                    self.fid.reset()
                    total_n_samples = 0

                    while total_n_samples < self.n_dataset_samples:
                        if pl_module.model.label_dim > 0:
                            labels = torch.randint(0, pl_module.model.label_dim, (self.sample_shape[0],))
                            labels = torch.nn.functional.one_hot(labels, num_classes=pl_module.model.label_dim).to(
                                pl_module.device)
                        else:
                            labels = None
                        samples = pl_module.sample(self.sample_shape, self.n_iters, use_ema=pl_module.use_ema, class_labels=labels).detach()
                        n_samples = samples.shape[0]

                        # check how many samples are left to reach our target number, if too many take a subset of the latest batch
                        if total_n_samples + n_samples > self.n_dataset_samples:
                            n_samples = self.n_dataset_samples - total_n_samples
                            samples = samples[:n_samples]

                        samples = rescaling_inv(samples.clamp(-1, 1))
                        samples = adjust_channels(samples)
                        self.fid.update(samples, real=False)
                        total_n_samples += n_samples

                    fid = self.fid.compute()
                    pl_module.model.train()
                if fid < self.best:
                    self.best = fid
                pl_module.log(f"best_FID_{self.n_iters}_iters", self.best, on_step=True, on_epoch=False, prog_bar=True,
                              logger=True)

                pl_module.log(f"FID_{self.n_iters}_iters", fid, on_step=True, on_epoch=False, prog_bar=True, logger=True)
