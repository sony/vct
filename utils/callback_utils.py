from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from custom_callbacks.fid_callback import FIDCallback
from custom_callbacks.generate_callback import GenerateCallback
from custom_callbacks.delete_checkpoints_callback import DeleteCheckpointsCallback


def get_delete_checkpoints_callback(cfg, path):
    return DeleteCheckpointsCallback(path, cfg.callback_log_frequency)

def get_callbacks(cfg: DictConfig):
    callbacks = []

    if cfg.log_samples:
        if cfg.model.use_ema:
            callbacks.append(
                GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=1, use_ema=True,
                                 every_n_iterations=cfg.callback_log_frequency, plot_type=cfg.dataset.plot_type))
            callbacks.append(
                GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=2, use_ema=True,
                                 every_n_iterations=cfg.callback_log_frequency, plot_type=cfg.dataset.plot_type))
        callbacks.append(
            GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=1, use_ema=False,
                             every_n_iterations=cfg.callback_log_frequency, plot_type=cfg.dataset.plot_type))
        callbacks.append(
            GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=2, use_ema=False,
                             every_n_iterations=cfg.callback_log_frequency, plot_type=cfg.dataset.plot_type))

    if cfg.compute_fid:
        callbacks.append(
            FIDCallback(tuple(cfg.dataset.fid_sample_shape), n_iters=1, n_dataset_samples=cfg.dataset.n_dataset_samples,
                        every_n_iterations=cfg.callback_log_frequency))
        callbacks.append(
            FIDCallback(tuple(cfg.dataset.fid_sample_shape), n_iters=2, n_dataset_samples=cfg.dataset.n_dataset_samples,
                        every_n_iterations=cfg.callback_log_frequency))
        if cfg.save_checkpoints:
            callbacks.append(ModelCheckpoint(every_n_train_steps=cfg.callback_log_frequency,
                                             save_top_k=1,
                                             monitor="FID_2_iters",
                                             mode="min",
                                             save_last=True,
                                             save_on_train_epoch_end=False,
                                             enable_version_counter=False,
                                             ))
    return callbacks
