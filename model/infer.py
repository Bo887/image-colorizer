import torch
import cv2

from dcgan_model import Generator
from data_loader import get_dataloader
from utils import lab_to_bgr

def infer(data_folder, params):
    data_loader = get_dataloader(data_folder, params["batch_size"])

    generator = Generator(1, 3)
    generator.load_state_dict(torch.load(params["gen_save_path"]))
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

params = {
    "batch_size" : 128,
    "epochs" : 100,
    "learning_rate" : 0.0002,
    "beta1" : 0.5,
    "print_interval": 10,
    "gen_save_path": "generator.pth",
    }

infer("../data/resized_color", params)
