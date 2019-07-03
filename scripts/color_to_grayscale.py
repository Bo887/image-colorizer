import os
import glob
import argparse
import cv2

parser = argparse.ArgumentParser(description = "Convert RGB images into grayscale for training.")
parser.add_argument("data_dir", type=str, help="location of the original RGB images")
parser.add_argument("output_dir", type=str, help="where to store the new grayscale images")
parser.add_argument("glob", type=str, help="glob to select the images")

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

for filename in data:
    image = cv2.imread(filename)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_filename = output_dir + os.path.basename(filename)
    print(filename + " -> " + new_filename)
    cv2.imwrite(new_filename, grayscale_image)
print("Done!")
