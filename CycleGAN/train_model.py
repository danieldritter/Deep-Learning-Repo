"""
Model for a Generative Adversarial Network trained to generate medical images
and their accompanying segmentation masks
"""
import image_processing
import torch.nn as nn
import model
import torch.utils as utils
from torchvision import transforms
import torch.optim as optim
import torch
from PIL import Image
import argparse

"""
TODO:
Data augmentations(rotations, affines, everything you got that's reasonable)
Documentation
Increase size of generator
retain graph issue
loss_graphing
"""


def train():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float,
                        help="Learning rate to use for training", default=.0002)
    parser.add_argument(
        "--show_images", help="shows generated image every 10 epochs", action="store_true")
    parser.add_argument("--checkpoint_frequency", type=int,
                        help="If given, saves a copy of the weights every x epochs, where x is the integer passed in. Default is no checkpoints saved")
    parser.add_argument(
        "--prev_model", help="if given, will load in previous saved model from a .tar file. Argument should be path to .tar file to load")
    parser.add_argument(
        "--path_to_A_images", help="path to folder containing nuclei images", default="./data/datasets/cezanne2photo/trainA")
    parser.add_argument(
        "--path_to_B_images", help="path to folder containing nuclei masks", default="./data/datasets/cezanne2photo/trainB")
    parser.add_argument("--batch_size", type=int,
                        help="size of batches to train on", default=1)
    parser.add_argument("--num_epochs", type=int,
                        help="number of epochs to train for", default=20)
    parser.add_argument("--lambda_weighting", type=int,
                        help="Weight to apply to cyclic consistency loss", default=10)
    parser.add_argument(
        "--show_progress", help="If passed, will store temp images generated from both generators after each training pass", action="store_true")
    args = parser.parse_args()

    # Defaults to using gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Creates datast and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    A_dataset = image_processing.CezanneDataset(
        args.path_to_A_images, transform)
    B_dataset = image_processing.CezanneDataset(
        args.path_to_B_images, transform)
    A_dataloader = utils.data.DataLoader(
        A_dataset, batch_size=args.batch_size, shuffle=True)
    B_dataloader = utils.data.DataLoader(
        B_dataset, batch_size=args.batch_size, shuffle=True)

    # Creates generators and discriminators
    image_gen = model.Generator(3, 3)
    mask_gen = model.Generator(3, 3)
    image_disc = model.Discriminator(3)
    mask_disc = model.Discriminator(3)

    # Add networks onto gpu
    image_gen.to(device)
    mask_gen.to(device)
    image_disc.to(device)
    mask_disc.to(device)

    cyclic_loss = nn.L1Loss()
    optimizer = optim.Adam(list(image_gen.parameters()) + list(mask_gen.parameters()) + list(
        image_disc.parameters()) + list(mask_disc.parameters()), lr=args.learning_rate)

    prev_epoch = 0
    # Loads in previous model if given
    if args.prev_model:
        checkpoint = torch.load(args.prev_model)
        image_gen.load_state_dict(checkpoint['image_gen_model'])
        mask_gen.load_state_dict(checkpoint['mask_gen_model'])
        image_disc.load_state_dict(checkpoint['image_disc_model'])
        mask_disc.load_state_dict(checkpoint['mask_disc_model'])
        optimizer.load_state_dict(checkpoint['optimizer_model'])
        prev_epoch = checkpoint['epoch']
        # TODO: Use loss here after adding in loss graphing

    for epoch in range(prev_epoch, args.num_epochs):
        print("Epoch: ", epoch)
        for i, batch in enumerate(zip(A_dataloader, B_dataloader)):
            # Puts inputs onto gpu
            image, mask = batch[0].to(device), batch[1].to(device)

            # Make predictions
            predicted_image = image_gen(mask)
            predicted_mask = mask_gen(image)
            im_discrim_prob = image_disc(image)
            mask_discrim_prob = mask_disc(mask)
            f_im_discrim_prob = image_disc(predicted_image)
            f_mask_discrim_prob = mask_disc(predicted_mask)
            recov_image = image_gen(predicted_mask)
            recov_mask = mask_gen(predicted_image)
            identity_image = image_gen(image)
            identity_mask = mask_gen(mask)

            # reshape probabilities for loss function
            im_discrim_prob = torch.t(im_discrim_prob)
            mask_discrim_prob = torch.t(mask_discrim_prob)
            f_im_discrim_prob = torch.t(f_im_discrim_prob)
            f_mask_discrim_prob = torch.t(f_mask_discrim_prob)
            # Get generator losses
            optimizer.zero_grad()
            im_to_mask_gen_loss = -torch.mean(
                torch.log(1 - f_im_discrim_prob[0]) + torch.log(im_discrim_prob[0]))
            mask_to_im_gen_loss = -torch.mean(
                torch.log(1 - f_mask_discrim_prob[0]) + torch.log(mask_discrim_prob[0]))
            # Get cyclic losses
            cyclic_loss_im_to_mask = cyclic_loss(recov_image, image)
            cyclic_loss_mask_to_im = cyclic_loss(recov_mask, mask)
            # Total up gen losses and optimize
            gen_loss = im_to_mask_gen_loss + mask_to_im_gen_loss + \
                args.lambda_weighting * \
                (cyclic_loss_im_to_mask + cyclic_loss_mask_to_im)

            # # Get discriminator losses
            im_discrim_loss = torch.mean(
                torch.log(1 - im_discrim_prob[0]) + torch.log(f_im_discrim_prob[0]))
            mask_discrim_loss = torch.mean(
                torch.log(1 - mask_discrim_prob[0]) + torch.log(f_mask_discrim_prob[0]))
            discrim_loss = im_discrim_loss + mask_discrim_loss

            identity_loss = args.lambda_weighting * \
                (cyclic_loss(identity_image, image) +
                 cyclic_loss(identity_mask, mask))

            total_loss = gen_loss + identity_loss
            total_loss.backward()
            optimizer.step()
            print("gen1_loss:", im_to_mask_gen_loss)
            print("gen2_loss:", mask_to_im_gen_loss)
            print("cyclic_loss_gen1", cyclic_loss_im_to_mask)
            print("cyclic_loss_gen2", cyclic_loss_mask_to_im)
            print("gen_loss", gen_loss)
            print("dis1_loss", im_discrim_loss)
            print("dis2_loss", mask_discrim_loss)
            print("dis_loss", discrim_loss)
            print("identity_loss", identity_loss)
            print("total_loss", total_loss)

            if args.show_progress:
                # A Image
                image = transforms.ToPILImage()(
                    predicted_image.cpu().detach()[0, :, :, :])
                image.save("./Images/temp_A.png")
                # B Image
                image = transforms.ToPILImage()(
                    predicted_mask.cpu().detach()[0, :, :, :])
                image.save("./Images/temp_B.png")

        if args.show_images:
            image = transforms.ToPILImage()(
                predicted_image.cpu().detach()[0, :, :, :])
            image.save("./Images/epoch_" + str(epoch) + ".png")
        # Saves a checkpoint if needed
        if args.checkpoint_frequency and epoch % args.checkpoint_frequency == 0:
            torch.save({
                'epoch': epoch,
                'gen_loss': gen_loss,
                'discrim_loss': discrim_loss,
                'image_gen_model': image_gen.state_dict(),
                'mask_gen_model': mask_gen.state_dict(),
                'image_disc_model': image_disc.state_dict(),
                'mask_disc_model': mask_disc.state_dict(),
                'optimizer_model': optimizer.state_dict()}, "./checkpoints/epoch_" + str(epoch) + ".tar")
    # Save last model after training
    torch.save({
        'epoch': epoch,
        'gen_loss': gen_loss,
        'discrim_loss': discrim_loss,
        'image_gen_model': image_gen.state_dict(),
        'mask_gen_model': mask_gen.state_dict(),
        'image_disc_model': image_disc.state_dict(),
        'mask_disc_model': mask_disc.state_dict(),
        'optimizer_model': optimizer.state_dict()}, "./checkpoints/epoch_" + str(epoch) + ".tar")


if __name__ == "__main__":
    train()
