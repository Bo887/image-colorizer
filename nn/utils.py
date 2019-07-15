import cv2
import torch
import matplotlib.pyplot as plt

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

def show_image(batch, num_images, orig_tensor, grayscale_tensor, predicted_tensor, save_path, headless=True):
    assert len(orig_tensor) == num_images
    assert len(grayscale_tensor) == num_images
    assert len(predicted_tensor) == num_images

    fig = plt.figure()
    for i in range(0, num_images):
        orig_image = lab_to_rgb(orig_tensor[i].permute(1, 2, 0).numpy())
        gray_image = grayscale_tensor[i].numpy().squeeze()
        predicted_image = lab_to_rgb(predicted_tensor[i].permute(1, 2, 0).detach().numpy())

        fig.add_subplot(num_images, 3, 3*i+1)
        plt.imshow(orig_image)
        fig.add_subplot(num_images, 3, 3*i+2)
        plt.imshow(gray_image, cmap="gray")
        fig.add_subplot(num_images, 3, 3*i+3)
        plt.imshow(predicted_image)
        plt.savefig(save_path + "batch_{}_predictions.png".format(batch))

    if not headless:
        plt.show()
