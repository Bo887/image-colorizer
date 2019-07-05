import torch
import torchvision
import os
import cv2

from dcgan_model import Generator
from dcgan_model import Discriminator
from data_loader import get_dataloader

params = {
    "batch_size" : 128,
    "epochs" : 10,
    "learning_rate" : 0.0002,
    "beta1" : 0.5,
    }

def train(data_folder, params):

    data_loader = get_dataloader(data_folder, params["batch_size"])

    generator = Generator(1, 3)
    discriminator = Discriminator(3, 1)

    generator.cuda()
    discriminator.cuda()

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=params["learning_rate"], betas=(params["beta1"], .999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=params["learning_rate"], betas=(params["beta1"], .999))

    real_label = 1
    fake_label = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(params["epochs"]):
        for i, images in enumerate(data_loader):

            l_images = images[:, 0, :, :]
            l_images = l_images.unsqueeze(1)
            batch_size = l_images.shape[0]
            l_images = torch.autograd.Variable(l_images.cuda())
            fake_images = generator(l_images)

train("../data/resized_color", params)
