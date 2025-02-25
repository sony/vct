import copy
from typing import Any, Optional

import lightning as L
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
import numpy as np
from utils.utils import power_function_beta

class LightningConsistencyModel(L.LightningModule):
    def __init__(self, cfg: DictConfig, model):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = model
        self.learning_rate = cfg.model.learning_rate
        self.ema_rate = cfg.model.ema_rate
        self.use_lr_decay = cfg.model.use_lr_decay
        self.lr_decay = cfg.model.lr_decay
        self.ema_type = cfg.model.ema_type
        self.use_ema = cfg.model.use_ema
        if self.use_ema:
            self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.evaluate_grad_var = cfg.evaluate_grad_var
        self.log_on_epoch = cfg.log_on_epoch
        self.log_on_step = cfg.log_on_epoch == False


    def configure_optimizers(self):
        if self.use_lr_decay:
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.learning_rate,
                                    weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.99))
        else:
            return torch.optim.RAdam(self.model.parameters(), lr=self.cfg.model.learning_rate,
                                     weight_decay=self.cfg.model.weight_decay)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            self.ema_update()

    @torch.no_grad()
    def ema_update(self):
        assert self.ema_type in ['traditional', 'power']
        if self.ema_type == 'traditional':
            ema_rate = self.ema_rate
        else:
            ema_rate = 1 - power_function_beta(std=self.ema_rate, t=self.global_step)
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_rate))

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            inputs = batch[0]
            labels = batch[1]
            if self.model.label_dim > 0:
                if self.cfg.dataset.name != 'imagenet':
                    labels = torch.nn.functional.one_hot(labels, num_classes=self.model.label_dim)
            else:
                labels = None
        else:
            inputs = batch
            labels = None
        if self.evaluate_grad_var:
            grad, (loss, log_dict) = torch.func.grad_and_value(self.model.loss, has_aux=True)(inputs, self.global_step, labels=labels)
            self.log("x_grad_variance", torch.var(grad.detach()), on_step=self.log_on_step, on_epoch=self.log_on_epoch,
                     prog_bar=True, logger=True)
        else:
            loss, log_dict = self.model.loss(inputs, self.global_step, labels=labels)
        for key, value in log_dict.items():
            self.log(key, value, on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def sample(self, sample_shape, n_iters, use_ema=True, class_labels=None, mid_t=[]):
        if use_ema:
            assert self.use_ema
            model = self.ema.eval()
        else:
            model = self.model
        return model.sample(sample_shape, n_iters, self.device, class_labels=class_labels, mid_t=mid_t)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.evaluate_grad_var:
            # Calculate the norm of the gradients
            grad_norm = 0.0
            for param in self.model.net.parameters():
                if param.grad is not None:
                    grad_norm += torch.norm(param.grad).item() ** 2  # Square of each parameter's norm
            grad_norm = grad_norm ** 0.5  # Square root to get total norm

            # Calculate the variance of the gradients
            grad_variance = 0.0
            grad_sum = 0.0
            grad_count = 0

            for param in self.model.net.parameters():
                if param.grad is not None:
                    grad_sum += torch.sum(param.grad)
                    grad_count += param.grad.numel()

            if grad_count > 0:  # Avoid division by zero
                mean = grad_sum / grad_count
                grad_variance = 0.0
                for param in self.model.net.parameters():
                    if param.grad is not None:
                        grad_variance += torch.sum((param.grad - mean) ** 2).item()
                grad_variance = grad_variance / grad_count
            self.log("grad_norm", grad_norm, on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=True, logger=True)
            self.log("grad_variance", grad_variance, on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=True, logger=True)

        if self.use_lr_decay:
            lr = self.learning_rate
            if self.global_step > 0:
                lr /= np.sqrt(max(self.global_step / self.lr_decay, 1))
            for g in optimizer.param_groups:
                g['lr'] = lr

        for param in self.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
