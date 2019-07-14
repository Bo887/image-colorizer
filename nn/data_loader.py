import torch
import torchvision
import cv2
import os
import numpy as np

from .utils import rgb_to_lab

class LABDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder):
        super().__init__()
        assert (os.path.exists(data_folder))
        dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=torchvision.transforms.ToTensor())

        self.preprocessed_images = []

        for image, _ in dataset:
            image = image.permute(1, 2, 0).numpy()
            lab_image = rgb_to_lab(image)
            torch_lab_image = torch.from_numpy(np.transpose(lab_image, (2, 0, 1)))
            self.preprocessed_images.append(torch_lab_image)

    def __len__(self):
        return len(self.preprocessed_images)

    def __getitem__(self, index):
        assert (index < len(self.preprocessed_images))
        return self.preprocessed_images[index]

def get_dataloader(data_folder, batch_size=128, shuffle=True, num_workers=0):
    dataset = LABDataset(data_folder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
