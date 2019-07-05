import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channels, out_channels, kernel_size=3, neg_slope=0.1, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(neg_slope)
    )

def deconv_layer(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Generator(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv1 = conv_layer(input_channels, 64)
        self.conv2 = conv_layer(64, 128)
        self.conv3 = conv_layer(128, 256)
        self.conv4 = conv_layer(256, 512)
        self.conv5 = conv_layer(512, 512)

        self.deconv1 = deconv_layer(512, 512)
        self.deconv2 = deconv_layer(512, 256)
        self.deconv3 = deconv_layer(256, 128)
        self.deconv4 = deconv_layer(128, 64)
        self.deconv5 = deconv_layer(64, output_channels)

    def forward(self, input):
        x = self.conv1(input)
        x1 = x
        x = self.conv2(x)
        x2 = x
        x = self.conv3(x)
        x3 = x
        x = self.conv4(x)
        x4 = x
        x = self.conv5(x)

        x = self.deconv1(x) 
        x += x4
        x = self.deconv2(x) 
        x += x3
        x = self.deconv3(x) 
        x += x2
        x = self.deconv4(x) 
        x += x1
        x = self.deconv5(x) 
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv_layer(in_channels, 64)
        self.conv2 = conv_layer(64, 128)
        self.conv3 = conv_layer(128, 256)
        self.conv4 = conv_layer(256, 512)
        self.conv5 = conv_layer(512, 512)
        self.conv6 = conv_layer(512, 512, kernel_size=7, stride=1, padding=0)
        self.conv7 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = F.sigmoid(x)
        return x
