from omegaconf import DictConfig

from datamodules.cifar10_datamodule import CIFAR10DataModule
from datamodules.mnist_datamodule import MNISTDataModule
from datamodules.fmnist_datamodule import FashionMNISTDataModule
from datamodules.imagenet_datamodule import ImagenetDataModule
from datamodules.ffhq_datamodule import FFHQDataModule

def get_datamodule(cfg: DictConfig):

    if cfg.dataset.name == 'cifar10':
        dm = CIFAR10DataModule(batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers,
                             data_dir=cfg.dataset.data_dir)

    elif cfg.dataset.name == 'mnist':
        dm = MNISTDataModule(batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers,
                             data_dir=cfg.dataset.data_dir)

    elif cfg.dataset.name == 'fashion_mnist':
        dm = FashionMNISTDataModule(batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers,
                             data_dir=cfg.dataset.data_dir)
    elif cfg.dataset.name == 'imagenet':
        dm = ImagenetDataModule(batch_size=cfg.dataset.batch_size, data_dir=cfg.dataset.data_dir, use_labels=cfg.model.class_conditional,
                                num_workers=cfg.dataset.num_workers)
    elif cfg.dataset.name == 'ffhq':
        dm = FFHQDataModule(batch_size=cfg.dataset.batch_size, data_dir=cfg.dataset.data_dir, use_labels=cfg.model.class_conditional,
                                num_workers=cfg.dataset.num_workers)
    else:
        raise NotImplementedError

    return dm


