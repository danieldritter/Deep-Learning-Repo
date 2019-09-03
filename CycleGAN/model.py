import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_c, out_c):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 3)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, 3)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, 4)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.up1 = nn.ConvTranspose2d(128, 256, 4)
        self.up1_bn = nn.BatchNorm2d(256)
        self.up2 = nn.ConvTranspose2d(256, 512, 3)
        self.up2_bn = nn.BatchNorm2d(512)
        self.up3 = nn.ConvTranspose2d(512, out_c, 3)

    def forward(self, input_tensor):
        out = F.relu(self.conv1(input_tensor))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.relu(self.up1(out))
        out = self.up1_bn(out)
        out = F.relu(self.up2(out))
        out = self.up2_bn(out)
        out = F.tanh(self.up3(out))
        return out


class Discriminator(nn.Module):

    def __init__(self, in_c):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 256, 4)
        self.conv1_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, 4)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 4)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(3904576, 2)

    def forward(self, input_tensor):
        out = F.leaky_relu(self.conv1(input_tensor))
        out = self.conv1_bn(out)
        out = F.leaky_relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.leaky_relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.softmax(self.fc1(out.view(out.size(0), -1)), dim=1)
        return out


def set_grad(model, state):
    """
    This is a function to turn off gradients for the discriminator
    while updating the generator
    """
    for param in model.parameters():
        param.requires_grad = state
