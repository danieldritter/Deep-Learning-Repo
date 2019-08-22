import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import numpy as np
import argparse
import data_processing


class Generator(nn.Module):

    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_size, 100, 4, stride=4)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.conv2 = nn.ConvTranspose2d(100, 90, 4, stride=4)
        self.conv2_bn = nn.BatchNorm2d(90)
        self.conv3 = nn.ConvTranspose2d(90, 80, 4, stride=4)
        self.conv3_bn = nn.BatchNorm2d(80)
        self.up1 = nn.ConvTranspose2d(80, 70, 4, stride=4)
        self.up1_bn = nn.BatchNorm2d(70)
        self.up2 = nn.ConvTranspose2d(70, 80, 4, stride=2)
        self.up2_bn = nn.BatchNorm2d(80)
        self.up3 = nn.ConvTranspose2d(80, 3, 4, stride=1)

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
        print(out.shape)
        return out


class Discriminator(nn.Module):

    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 40, 2)
        self.conv1_bn = nn.BatchNorm2d(40)
        self.conv2 = nn.Conv2d(40, 30, 2)
        self.conv2_bn = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, 20, 2)
        self.conv3_bn = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(5283920, 2)

    def forward(self, input_tensor):
        out = F.leaky_relu(self.conv1(input_tensor))
        out = self.conv1_bn(out)
        out = F.leaky_relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.leaky_relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.softmax(self.fc1(out.view(out.size(0), -1)), dim=1)
        return out


def __main__():
    parser = argparse.ArgumentParser(description="A Program to automatically")
    parser.add_argument("--batch_size", type=int,
                        help="Batch Size to use when training", default=2)
    parser.add_argument("--learning_rate", type=float,
                        help="Learning Rate to use when training, float between [0,1)", default=.0002)
    parser.add_argument(
        "--path_to_data", help="file path to folder containing card arts", default="D:/mtg_card_data")
    parser.add_argument("--num_epochs", type=int,
                        help="number of epochs to train for", default=100)
    parser.add_argument(
        "--show_images", help="shows generated image every 10 epochs", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.Resize((517, 517)), transforms.ToTensor()])
    dataset = data_processing.CardDataset(
        args.path_to_data, transform=transform)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
    gen = Generator(100)
    dis = Discriminator(3)
    gen_optimizer = optim.Adam(gen.parameters(), lr=.002)
    dis_optimizer = optim.Adam(dis.parameters(), lr=.00001)
    loss = nn.BCELoss()

    gen.to(device)
    dis.to(device)

    for epoch in range(args.num_epochs):
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            dis_optimizer.zero_grad()
            noise = torch.randn((args.batch_size, 100, 1, 1)).to(device)
            fake_batch = gen(noise)
            discrim_prob_real = dis(batch)
            discrim_prob_fake = dis(fake_batch)
            discrim_loss_real = -torch.sum(torch.log(discrim_prob_real[:, 0]))
            discrim_loss_fake = -torch.sum(
                torch.log(1 - discrim_prob_fake[:, 0]))
            discrim_loss = discrim_loss_real + discrim_loss_fake
            print(discrim_loss)
            discrim_loss.backward()
            dis_optimizer.step()

            gen_optimizer.zero_grad()
            noise = torch.randn((args.batch_size, 100, 1, 1)).to(device)
            discrim_prob_fake = dis(gen(noise))
            gen_loss = -torch.sum(torch.log(discrim_prob_fake[:, 0]))
            print(gen_loss)
            gen_loss.backward()
            gen_optimizer.step()

            if args.show_images:
                image = transforms.ToPILImage()(
                    fake_batch.cpu().detach()[0, :, :, :])
                image.save("./Images/epoch_" + str(epoch) + ".png")
                image = transforms.ToPILImage()(batch.cpu()[0, :, :, :])
                image.save("./Images/real_epoch_" + str(epoch) + ".png")


if __name__ == "__main__":
    __main__()
