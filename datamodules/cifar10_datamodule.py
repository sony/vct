import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from utils.utils import ResumableDataLoader


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, data_dir: str = "./"):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.fid = CIFAR10(self.data_dir, train=True, transform=transforms.ToTensor())
            self.test = CIFAR10(self.data_dir, train=False, transform=self.transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self, shuffle=True):
        return ResumableDataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle,
                                   num_workers=self.num_workers)

    def fid_dataloader(self):
        return DataLoader(self.fid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
