import torch
import cv2

from dcgan_model import Generator
from data_loader import get_dataloader

def infer(data_folder, params):
    data_loader = get_dataloader(data_folder, params["batch_size"])

    generator = Generator(1, 3)
    generator.load_state_dict(torch.load(params["gen_save_path"]))
    generator.eval()

    for i, images in enumerate(data_loader):
        grayscale_images = images[:, 0:1, :, :]
        predicted_images = generator(grayscale_images)
        orig_image = images[0].permute(1, 2, 0).numpy()
        orig_image *= 255
        orig_image[:, :, 2] -= 128
        orig_image[:, :, 1] -= 128
        orig_image[:, :, 0] /= 255/100
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_LAB2BGR)
        predicted_img = predicted_images[0].permute(1, 2, 0).detach().numpy()
        predicted_img *= 255
        predicted_img[:, :, 2] -= 128
        predicted_img[:, :, 1] -= 128
        predicted_img[:, :, 0] /= 255/100
        predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_LAB2BGR)
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
    "gen_save_path": "generator.model",
    "disc_save_path": "discriminator.model"
    }

infer("../data/resized_color", params)
