import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import numpy as np
import argparse


class Generator(nn.Module):

    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            input_size, 512, 6, stride=2, bias=False)
        self.conv1_bn = nn.InstanceNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, stride=1, bias=False)
        self.conv2_bn = nn.InstanceNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 2, stride=1, bias=False)
        self.conv3_bn = nn.InstanceNorm2d(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.up1_bn = nn.InstanceNorm2d(64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.up2_bn = nn.InstanceNorm2d(32)
        self.up3 = nn.ConvTranspose2d(32, 3, 2, stride=2, bias=False)

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

    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            input_size, 512, 5, stride=1, bias=False)
        self.conv1_bn = nn.InstanceNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, 8, stride=1, bias=False)
        self.conv2_bn = nn.InstanceNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, 2, stride=1, bias=False)
        self.conv3_bn = nn.InstanceNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 2, stride=2, bias=False)
        self.conv4_bn = nn.InstanceNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 2, stride=2, bias=False)
        self.conv5_bn = nn.InstanceNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 2, stride=2, bias=False)
        self.conv6_bn = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(2048, 2)

    def forward(self, input_tensor):
        out = F.leaky_relu(self.conv1(input_tensor))
        out = self.conv1_bn(out)
        out = F.leaky_relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.leaky_relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.leaky_relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = F.leaky_relu(self.conv5(out))
        out = self.conv5_bn(out)
        out = F.leaky_relu(self.conv6(out))
        out = self.conv6_bn(out)
        out = F.softmax(self.fc1(out.view(out.size(0), -1)), dim=1)
        return out


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int,
                        help="Batch Size to use when training", default=8)
    parser.add_argument("--learning_rate", type=float,
                        help="Learning Rate to use when training, float between [0,1)", default=.0002)
    parser.add_argument(
        "--path_to_data", help="file path to folder containing images", default="data")
    parser.add_argument("--num_epochs", type=int,
                        help="number of epochs to train for", default=100)
    parser.add_argument(
        "--show_images", help="shows generated image after every epoch", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")
    transform = transforms.Compose(
        [transforms.Resize((80, 80)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.path_to_data, transform=transform)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
    gen = Generator(100)
    dis = Discriminator(3)
    gen_optimizer = optim.Adam(gen.parameters(), lr=.001)
    dis_optimizer = optim.Adam(dis.parameters(), lr=.0001)
    gen_lr_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, 5)
    loss = nn.BCELoss()

    gen.to(device)
    dis.to(device)

    for epoch in range(args.num_epochs):
        for i, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            dis_optimizer.zero_grad()
            noise = torch.randn((args.batch_size, 100, 1, 1)).to(device)
            fake_batch = gen(noise)
            # Gets probabilities on real and fake images, then calculates losses
            discrim_prob_real = dis(batch)
            discrim_prob_fake = dis(fake_batch)
            discrim_loss_real = -torch.sum(torch.log(discrim_prob_real[:, 0]))
            discrim_loss_fake = -torch.sum(
                torch.log(1 - discrim_prob_fake[:, 0]))
            discrim_loss = .5 * (discrim_loss_real + discrim_loss_fake)
            print(discrim_loss)
            discrim_loss.backward()
            dis_optimizer.step()
            gen_optimizer.zero_grad()
            # Generates another fake batch of images to update generator
            noise = torch.randn((args.batch_size, 100, 1, 1)).to(device)
            discrim_prob_fake = dis(gen(noise))
            gen_loss = -torch.sum(torch.log(discrim_prob_fake[:, 0]))
            print(gen_loss)
            gen_loss.backward()
            gen_optimizer.step()

        # Saves images to a directory if command line argument is passed
        if args.show_images:
            image = transforms.ToPILImage()(
                fake_batch.cpu().detach()[0, :, :, :])
            image.save("./Images/epoch_" + str(epoch) + ".png")
            image = transforms.ToPILImage()(batch.cpu()[0, :, :, :])
            image.save("./Images/real_epoch_" + str(epoch) + ".png")
        gen_lr_scheduler.step()


if __name__ == "__main__":
    __main__()
