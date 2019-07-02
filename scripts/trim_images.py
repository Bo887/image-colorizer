import os
import glob
import argparse
import random

parser = argparse.ArgumentParser(description = "Randomly delete extra images from a folder")
parser.add_argument("data_dir", type=str, help="location of the images")
parser.add_argument("glob", type=str, help="glob to select the images")
parser.add_argument("num_to_keep", type=int, help="number of images to keep")

args = parser.parse_args()

num_to_keep = args.num_to_keep

data_dir = args.data_dir
if data_dir[-1] != "/":
    data_dir += "/"

data = sorted(glob.glob(data_dir + args.glob))
num_pictures = len(data)

assert(num_to_keep <= num_pictures)

num_to_delete = num_pictures - num_to_keep

for i in range(num_to_delete):
    rand_idx = random.randint(0, num_pictures-1)
    to_delete = data[rand_idx]
    while not os.path.exists(to_delete):
        rand_idx = random.randint(0, num_pictures-1)
        to_delete = data[rand_idx]
    print("Deleting {}".format(data[rand_idx]))
    os.unlink(data[rand_idx])
print("Done!")
