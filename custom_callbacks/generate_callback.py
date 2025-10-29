import PIL.Image
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch import seed_everything


class GenerateCallback(L.Callback):

    def __init__(self, sample_shape, n_iters, use_ema, every_n_iterations=1, plot_type='grid'):
        super().__init__()
        assert [np.sqrt(sample_shape[0]) == np.floor(np.sqrt(sample_shape[0]))]
        self.sample_shape = sample_shape
        self.side_size = int(np.sqrt(sample_shape[0]))
        self.every_n_iterations = every_n_iterations  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.n_iters = n_iters
        self.plot_type = plot_type
        self.use_ema = use_ema

    def log_image_grid(self, img, drange, grid_size, iter, trainer):
        lo, hi = drange
        img = np.asarray(img.cpu(), dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

        gw, gh = grid_size
        _N, C, H, W = img.shape
        img = img.reshape(gh, gw, C, H, W)
        img = img.transpose(0, 3, 1, 4, 2)
        img = img.reshape(gh * H, gw * W, C)

        assert C in [1, 3]
        key = f"samples {iter} iters"
        if self.use_ema:
            key += " ema"

        if C == 1:
            fig = PIL.Image.fromarray(img[:, :, 0], 'L')
        if C == 3:
            fig = PIL.Image.fromarray(img, 'RGB')
        trainer.logger.log_image(key=key, images=[fig], step=trainer.global_step)

    def log_samples(self, samples, iter, trainer):
        samples = samples.cpu()
        if self.plot_type == 'grid':
            fig = torchvision.utils.make_grid(samples.clamp(-1, 1), nrow=4, value_range=(-1, 1))
        elif self.plot_type == 'scatter':
            fig, ax = plt.subplots()
            ax.scatter(samples[:, 0], samples[:, 1], s=1)
            title = f'Samples {iter} iters'
            if self.use_ema:
                title += f' ema'
            ax.set_title(title)
            plt.close(fig)
        else:
            raise NotImplementedError
        key = f"samples {iter} iters"
        if self.use_ema:
            key += " ema"
        trainer.logger.log_image(key=key, images=[fig], step=trainer.global_step)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with isolate_rng():
            torch.manual_seed(32)
            if pl_module.model.label_dim > 0:
                labels = torch.arange(self.side_size).repeat_interleave(self.side_size).view(-1)
                if self.side_size > pl_module.model.label_dim:
                    labels = torch.where(labels < pl_module.model.label_dim, labels, labels - (pl_module.model.label_dim))
                labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=pl_module.model.label_dim).to(pl_module.device)
            else:
                labels = None
            # Reconstruct images
            with torch.no_grad():
                pl_module.model.eval()
                samples = pl_module.sample(self.sample_shape, self.n_iters, self.use_ema, class_labels=labels)
                pl_module.model.train()

            if self.plot_type == 'scatter':
                self.log_samples(samples, iter=self.n_iters, trainer=trainer)
            else:
                # Plot and add to tensorboard
                self.log_image_grid(samples, drange=[-1, 1], iter=self.n_iters, trainer=trainer, grid_size=(self.side_size, self.side_size))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_iterations == 0:
            with isolate_rng():
                torch.manual_seed(32)
                if pl_module.model.label_dim > 0:
                    labels = torch.arange(self.side_size).repeat_interleave(self.side_size).view(-1)
                    if self.side_size > pl_module.model.label_dim:
                        labels = torch.where(labels < pl_module.model.label_dim, labels,
                                             labels - (pl_module.model.label_dim))
                    labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=pl_module.model.label_dim).to(
                        pl_module.device)
                else:
                    labels = None

                # Reconstruct images
                with torch.no_grad():
                    pl_module.model.eval()
                    samples = pl_module.sample(self.sample_shape, self.n_iters, self.use_ema, class_labels=labels)
                    pl_module.model.train()

                # Plot and add to tensorboard
                if self.plot_type == 'scatter':
                    self.log_samples(samples, iter=self.n_iters, trainer=trainer)
                else:
                    self.log_image_grid(samples, drange=[-1, 1], iter=self.n_iters, trainer=trainer, grid_size=(self.side_size, self.side_size))
