import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable

import os

from dcgan_model import Generator
from dcgan_model import Discriminator
from data_loader import get_dataloader

from utils import save_model

def train(data_folder, params):
    data_loader = get_dataloader(data_folder, params["batch_size"])

    # generator takes in a single channel image and outputs a 3-channel image
    generator = Generator(1, 3)
    # discriminator takes in a 3-channel image a single value
    discriminator = Discriminator(3, 1)

    generator.cuda()
    discriminator.cuda()

    g_optim = optim.Adam(generator.parameters(), lr=params["learning_rate"], betas=(params["beta1"], .999))
    d_optim = optim.Adam(discriminator.parameters(), lr=params["learning_rate"], betas=(params["beta1"], .999))

    d_criterion = torch.nn.BCEWithLogitsLoss()
    g_adv_criterion = torch.nn.BCEWithLogitsLoss()
    g_dist_criterion = torch.nn.L1Loss()

    save_path = params["save_path"]
    if not save_path[-1] == "/":
        save_path += "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # for each epoch
    for epoch in range(params["epochs"]):

        # for each batch
        for i, images in enumerate(data_loader):

            d_loss, g_loss = single_iteration(images, generator, discriminator, g_optim, d_optim, g_adv_criterion, g_dist_criterion, d_criterion)

        if epoch % params["print_interval"] == 0:
            print("EPOCH {0}:\tD-Loss: {1:.4f}\tG-Loss: {2:.4f}".format(epoch, d_loss.item(), g_loss.item()))

        if "save_interval" in params and epoch % params["save_interval"] == 0:
            filename = save_path + "model_epoch_{}.pth".format(epoch)
            save_model(filename, epoch, generator, discriminator, g_optim, d_optim)

    save_model(save_path + "model_final.pth", epoch, generator, discriminator, g_optim, d_optim)

def single_iteration(images, generator, discriminator, g_optim, d_optim, g_adv_criterion, g_dist_criterion, d_criterion):
    # get the corresponding grayscale images
    grayscale_images = images[:, 0:1, :, :]
    grayscale_images, images = Variable(grayscale_images.cuda()), Variable(images.cuda())

    # train the discriminator on real color images
    discriminator.zero_grad()
    real_predictions = discriminator(images)
    real_labels = torch.FloatTensor(images.size(0)).fill_(1)
    real_labels = Variable(real_labels.cuda())

    d_real_loss = d_criterion(torch.squeeze(real_predictions), real_labels)
    d_real_loss.backward()

    # train the discriminator on fake color images that are generated from the grayscale images
    fake_images = generator(grayscale_images)
    fake_predictions = discriminator(fake_images.detach())
    fake_labels = torch.FloatTensor(fake_images.size(0)).fill_(1)
    fake_labels = Variable(fake_labels.cuda())
    d_fake_loss = d_criterion(torch.squeeze(fake_predictions), fake_labels)
    d_fake_loss.backward()

    total_d_loss = d_real_loss + d_fake_loss
    d_optim.step()

    # train the generator using the discriminator's predictions
    generator.zero_grad()
    fake_predictions = discriminator(fake_images)
    g_adversarial_loss = g_adv_criterion(torch.squeeze(fake_predictions), real_labels)
    g_dist_loss = g_dist_criterion(fake_images.view(fake_images.size(0), -1), images.view(images.size(0), -1))
    total_g_loss = g_adversarial_loss + 100*g_dist_loss
    total_g_loss.backward()
    g_optim.step()
    return total_d_loss, total_g_loss

params = {
    "batch_size" : 128,
    "epochs" : 1000,
    "learning_rate" : 0.009,
    "beta1" : 0.5,
    "print_interval": 10,
    "save_interval": 100,
    "save_path": "models/"
    }

train("../data/resized_color", params)
