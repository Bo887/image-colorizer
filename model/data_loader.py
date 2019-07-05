import torch
import torchvision
import cv2
import os
import numpy as np

class LABDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder):
        super().__init__()
        assert (os.path.exists(data_folder))
        dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=torchvision.transforms.ToTensor())

        self.preprocessed_images = []

        for image, _ in dataset:
            image = image.permute(1, 2, 0).numpy()
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab_image[:, :, 0] *= 255/100
            lab_image[:, :, 1] += 128
            lab_image[:, :, 2] += 128
            lab_image /= 255
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
