import os
import glob
import argparse

parser = argparse.ArgumentParser(description = "Rename image files from COCO into number format (i.e. 0001.jpg, 1234.jpg).")
parser.add_argument("data_dir", type=str, help="location of the images")
parser.add_argument("glob", type=str, help="glob to select the images")

args = parser.parse_args()

data_dir = args.data_dir
if data_dir[-1] != "/":
    data_dir += "/"

data = sorted(glob.glob(data_dir + args.glob))
num_digits = len(str(len(data)))

format_str = "{:0" + str(num_digits) + "d}.jpg"

current_num = 0
for filename in data:
    new_name = data_dir + format_str.format(current_num)
    print(filename + " -> " + new_name)
    current_num += 1
    os.rename(filename, new_name)
print("Done!")
