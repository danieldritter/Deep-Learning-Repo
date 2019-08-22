import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_c, out_c):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 160, 3)
        self.conv1_bn = nn.BatchNorm2d(160)
        self.conv2 = nn.Conv2d(160, 140, 3)
        self.conv2_bn = nn.BatchNorm2d(140)
        self.conv3 = nn.Conv2d(140, 120, 4)
        self.conv3_bn = nn.BatchNorm2d(120)
        self.up1 = nn.ConvTranspose2d(120, 140, 4)
        self.up1_bn = nn.BatchNorm2d(140)
        self.up2 = nn.ConvTranspose2d(140, 160, 3)
        self.up2_bn = nn.BatchNorm2d(160)
        self.up3 = nn.ConvTranspose2d(160, out_c, 3)

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
        self.conv1 = nn.Conv2d(in_c, 80, 4)
        self.conv1_bn = nn.BatchNorm2d(80)
        self.conv2 = nn.Conv2d(80, 60, 4)
        self.conv2_bn = nn.BatchNorm2d(60)
        self.conv3 = nn.Conv2d(60, 40, 4)
        self.conv3_bn = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(2440360, 2)

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
