import os
import glob
import argparse
import cv2

parser = argparse.ArgumentParser(description = "Resize batch of images.")
parser.add_argument("data_dir", type=str, help="location of the original RGB images")
parser.add_argument("output_dir", type=str, help="where to store the new grayscale images")
parser.add_argument("glob", type=str, help="glob to select the images")
parser.add_argument("height", type=int, help="new height of images")
parser.add_argument("width", type=int, help="new width of images")

args = parser.parse_args()

data_dir = args.data_dir
assert (os.path.exists(data_dir))
if data_dir[-1] != "/":
    data_dir += "/"

output_dir = args.output_dir
assert (os.path.exists(output_dir))
if output_dir[-1] != "/":
    output_dir += "/"

data = sorted(glob.glob(data_dir + args.glob))

new_shape = (args.height, args.width)

for filename in data:
    image = cv2.imread(filename)
    new_image = cv2.resize(image, new_shape)
    new_filename = output_dir + os.path.basename(filename)
    print(filename + " -> " + new_filename)
    cv2.imwrite(new_filename, new_image)
print("Done!")
