import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from utils.utils import ResumableDataLoader
from utils.utils import rescaling


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, data_dir: str = "./"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), rescaling])

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.fid = FashionMNIST(self.data_dir, train=True, transform=transforms.ToTensor())
            self.test = FashionMNIST(self.data_dir, train=False, transform=self.transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test = FashionMNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict = FashionMNIST(self.data_dir, train=False, transform=self.transform)

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
