import cv2
import torch

def rgb_to_lab(rgb_image):
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] *= 255/100
    lab_image[:, :, 1] += 128
    lab_image[:, :, 2] += 128
    lab_image /= 255
    return lab_image

def lab_to_rgb(lab_image):
    lab_image *= 255
    lab_image[:, :, 0] /= (255/100)
    lab_image[:, :, 1] -= 128
    lab_image[:, :, 2] -= 128
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image

def lab_to_bgr(lab_image):
    rgb_image = lab_to_rgb(lab_image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

def save_model(filename, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer):
    state_dict = {
        "epoch": epoch,
        "generator_state": generator.state_dict(),
        "generator_optimizer_state": generator_optimizer.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "discriminator_optimizer_state": discriminator_optimizer.state_dict()
    }

    torch.save(state_dict, filename)
