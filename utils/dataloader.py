from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import os


class VQVAEDataset(Dataset):
    def __init__(self, data):
        super(VQVAEDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        data, label = self.data[index]
        return data,

    def __len__(self):
        return len(self.data)
        pass


def Dataloader(path, batch_size=128, num_workers=16, shuffle=True):
    """
    Reads image folder and returns data in batches.
    """
    if path == "":
        return None

    def normalize(image):
        return torch.div(torch.sub(image, torch.min(image)), torch.max(image) - torch.min(image))

    dataset = ImageFolder(
        os.path.dirname(os.path.dirname(path)),
        transform=Compose([ToTensor(), normalize])
    )
    dataset = VQVAEDataset(dataset)
    dataset = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return dataset

