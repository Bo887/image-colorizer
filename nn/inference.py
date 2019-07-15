import torch
import os

from .dcgan_model import Generator
from .data_loader import get_dataloader
from .utils import lab_to_rgb
from .utils import show_image

def infer(data_folder, params):
    data_loader = get_dataloader(data_folder, params["batch_size"])

    generator = Generator(1, 3)
    generator.load_state_dict(torch.load(params["model_path"])["generator_state"])
    generator.eval()

    save_path = params["predictions_save_path"]
    if save_path[-1] != "/":
        save_path += "/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, images in enumerate(data_loader):
        grayscale_images = images[:, 0:1, :, :]
        predicted_images = generator(grayscale_images)

        show_image(i, len(images), images, grayscale_images, predicted_images, save_path, params["headless"])
