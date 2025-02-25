import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.utils import ResumableDataLoader
from utils.dataset import ImageFolderDataset


class FFHQDataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_dir, use_labels, xflip=False, cache=True, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.use_labels = use_labels
        self.xflip = xflip
        self.cache = cache
        self.transform = transforms.Compose([
            #.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = ImageFolderDataset(path=self.data_dir, use_labels=self.use_labels, xflip=self.xflip, cache=False, transform=self.transform)
            self.fid = ImageFolderDataset(path=self.data_dir, use_labels=self.use_labels, xflip=False, cache=False, transform=transforms.ToTensor())

    def train_dataloader(self, shuffle=True):
        return ResumableDataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle,
                                   num_workers=self.num_workers)
    def fid_dataloader(self):
        return DataLoader(self.fid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
