import torch
import cv2

from dcgan_model import Generator
from data_loader import get_dataloader
from utils import lab_to_bgr

def infer(data_folder, params):
    data_loader = get_dataloader(data_folder, params["batch_size"])

    generator = Generator(1, 3)
    generator.load_state_dict(torch.load(params["save_path"])["generator_state"])
    generator.eval()

    for i, images in enumerate(data_loader):
        grayscale_images = images[:, 0:1, :, :]
        predicted_images = generator(grayscale_images)
        orig_image = lab_to_bgr(images[1].permute(1, 2, 0).numpy())
        predicted_img = lab_to_bgr(predicted_images[1].permute(1, 2, 0).detach().numpy())
        while True:
            cv2.imshow("orig_image", orig_image) 
            cv2.imshow("color", predicted_img)
            cv2.waitKey(30)
