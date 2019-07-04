import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, params):
        super().__init__()

        # Z -> ngf*8 x 4 x 4
        input_dim, output_dim = params["nz"], params["ngf"]*8
        self.deconv1 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(output_dim)

        # ngf*8 x 4 x 4 -> ngf*4 x 8 x 8
        input_dim, output_dim = params["ngf"]*8, params["ngf"]*4
        self.deconv2 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(output_dim)

        # ngf*4 x 8 x 8 -> ngf*2 x 16 x 16
        input_dim, output_dim = params["ngf"]*4, params["ngf"]*2
        self.deconv3 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(output_dim)

        # ngf*2 x 16 x 16 -> ngf x 32 x 32
        input_dim, output_dim = params["ngf"]*2, params["ngf"]
        self.deconv4 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(output_dim)

        # ngf x 32 x 32 -> nc x 64 x 64 (output)
        input_dim, output_dim = params["ngf"], params["nc"]
        self.deconv5 = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, inputs):
        x = F.relu(self.batchnorm1(self.deconv1(inputs)))
        x = F.relu(self.batchnorm2(self.deconv2(x)))
        x = F.relu(self.batchnorm3(self.deconv3(x)))
        x = F.relu(self.batchnorm4(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x

class Discriminator(nn.Module):

    def __init__(self, params):
        super().__init__()

        # nc x 64 x 64 (output of Generator) ->  ndf x 32 x 32
        input_dim, output_dim = params["nc"], params["ndf"]
        self.conv1 = nn.Conv2d(input_dim, output_dim, 4, 2, 1, bias=False)

        # ndf x 32 x 32 -> ndf*2 x 16 x 16
        input_dim, output_dim = params["ndf"], params["ndf"]*2
        self.conv2 = nn.Conv2d(input_dim, output_dim, 4, 2, 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(output_dim)

        # ndf*2 x 16 x 16 -> ndf*4 x 8 x 8
        input_dim, output_dim = params["ndf"]*2, params["ndf"]*4
        self.conv3 = nn.Conv2d(input_dim, output_dim, 4, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(output_dim)

        # ndf*4 x 8 x 8 -> ndf*8 x 4 x 4
        input_dim, output_dim = params["ndf"]*4, params["ndf"]*8
        self.conv4 = nn.Conv2d(input_dim, output_dim, 4, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(output_dim)

        # ndf*8 x 4 x 4 -> sigmoid (0-1)
        input_dim, output_dim = params["ndf"]*8, 1
        self.conv5 = nn.Conv2d(input_dim, output_dim, 4, 1, 0, bias=False)

    def forward(self, inputs):
        x = F.leaky_relu(self.conv1(inputs), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.batchnorm4(self.conv4(x)), negative_slope=0.2, inplace=True)
        x = F.sigmoid(self.conv5(x))
        return x

"""
params = {
    "bsize" : 128,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 10,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2}# Save step.
g = Generator(params)
d = Discriminator(params)
"""
