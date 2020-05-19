import argparse
import torch
import model
import os
from PIL import Image
from torchvision import transforms


def __main__():
    """
    This script is to visualize the images output by the trained CycleGAN
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", help="Path to model checkpoint to load in for inference",
                        default="saved_models/demo_checkpoint.tar")
    parser.add_argument(
        "-direction", help="One of 'im2mask' or 'mask2im', determines which generator to pass input images through", default='im2mask')
    parser.add_argument(
        '-image_directory', help="filepath to directory containing images to translate", default="./data/datasets/cezanne2photo/testB")
    args = parser.parse_args()
    checkpoint = torch.load(args.path_to_model, map_location="cpu")
    saved_model = model.Generator(3, 3)

    # Loads the corresponding generator depending on which direction we are putting images through
    if args.direction == 'im2mask':
        saved_model.load_state_dict(checkpoint["mask_gen_model"])
    if args.direction == "mask2im":
        saved_model.load_state_dict(checkpoint["image_gen_model"])

    # Makes a directory to store the output images in
    if not os.path.exists(args.image_directory + "/predictions"):
        os.makedirs(args.image_directory + "/predictions")
    print(os.listdir(args.image_directory))

    # Loops through all the images, transforms them, puts them through the model, and then saves the output 
    for file in os.listdir(args.image_directory)[:1]:
        if not os.path.isdir(args.image_directory + "/" + file):
            im = transforms.Resize((450, 450))(
                Image.open(args.image_directory + "/" + file))
            input = transforms.ToTensor()(im)
            input = input.view(1, 3, 450, 450)
            prediction = saved_model(input).squeeze()
            im = transforms.ToPILImage()(prediction)
            im.save(args.image_directory + "/predictions/" + file)


if __name__ == "__main__":
    __main__()
