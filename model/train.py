import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable

from dcgan_model import Generator
from dcgan_model import Discriminator
from data_loader import get_dataloader

def train(data_folder, params):
    data_loader = get_dataloader(data_folder, params["batch_size"])

    # generator takes in a single channel image and outputs a 3-channel image
    generator = Generator(1, 3)
    # discriminator takes in a 3-channel image a single value
    discriminator = Discriminator(3, 1)

    generator.cuda()
    discriminator.cuda()

    generator_optimizer = optim.Adam(generator.parameters(), lr=params["learning_rate"], betas=(params["beta1"], .999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=params["learning_rate"], betas=(params["beta1"], .999))

    discriminator_criterion = torch.nn.BCEWithLogitsLoss()
    generator_adversarial_criterion = torch.nn.BCEWithLogitsLoss()
    generator_dist_criterion = torch.nn.L1Loss()

    # for each epoch
    for epoch in range(params["epochs"]):

        # for each batch
        for i, images in enumerate(data_loader):

            # get the corresponding grayscale images
            grayscale_images = images[:, 0:1, :, :]
            grayscale_images, images = Variable(grayscale_images.cuda()), Variable(images.cuda())

            # train the discriminator on real color images
            discriminator.zero_grad()
            real_predictions = discriminator(images)
            real_labels = torch.FloatTensor(images.size(0)).fill_(1)
            real_labels = Variable(real_labels.cuda())

            discriminator_real_loss = discriminator_criterion(torch.squeeze(real_predictions), real_labels)
            discriminator_real_loss.backward()

            # train the discriminator on fake color images that are generated from the grayscale images
            fake_images = generator(grayscale_images)
            fake_predictions = discriminator(fake_images.detach())
            fake_labels = torch.FloatTensor(fake_images.size(0)).fill_(1)
            fake_labels = Variable(fake_labels.cuda())
            discriminator_fake_loss = discriminator_criterion(torch.squeeze(fake_predictions), fake_labels)
            discriminator_fake_loss.backward()

            total_discriminator_loss = discriminator_real_loss + discriminator_fake_loss
            discriminator_optimizer.step()

            # train the generator using the discriminator's predictions
            generator.zero_grad()
            fake_predictions = discriminator(fake_images)
            generator_adversarial_loss = generator_adversarial_criterion(torch.squeeze(fake_predictions), real_labels)
            generator_dist_loss = generator_dist_criterion(fake_images.view(fake_images.size(0), -1), images.view(images.size(0), -1))
            total_generator_loss = generator_adversarial_loss + 100*generator_dist_loss
            generator_optimizer.step()

            if epoch % params["print_interval"] == 0:
                print("D-Loss: {0:.4f}\tG-Loss: {1:.4f}".format(total_discriminator_loss.item(), total_generator_loss.item()))

    torch.save(generator.state_dict(), params["gen_save_path"])
    torch.save(discriminator.state_dict(), params["disc_save_path"])

params = {
    "batch_size" : 128,
    "epochs" : 1000,
    "learning_rate" : 0.0002,
    "beta1" : 0.5,
    "print_interval": 10,
    "gen_save_path": "generator.model",
    "disc_save_path": "discriminator.model"
    }

train("../data/resized_color", params)
