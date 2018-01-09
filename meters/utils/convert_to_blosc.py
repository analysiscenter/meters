"""File to compless images to blosc format.
Can use with aruments from command line.
-i - path_from. It's a path with images.
-b - path_to. It's a path with created blosc files.
-d delete all images
If path_from and path_to is empty - images will getting from working directory and blosc files saved in the same place"""
import os
import sys
import dill

import blosc
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def compress():
    """Convert images from any format to blosc format."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_images', type=str, help='name of path with images')
    parser.add_argument('-b', '--path_blosc', type=str, help='name of path where blosc files will be created')
    parser.add_argument('-d', '--delete', action='store_true', help='delete images')
    args = parser.parse_args()

    path_from = args.path_images if args.path_images else './'
    path_to = args.path_blosc if args.path_blosc else path_from

    print('path from: %s'%path_from)
    print('path to: %s\n'%path_to)

    files_name = os.listdir(path_from)
    for imfile in tqdm(files_name):
        impath = os.path.join(path_from, imfile)
        if not os.path.isdir(impath):
            image = plt.imread(impath)
            image_path = os.path.join(path_to, imfile[:-4])
            with open(image_path + '.blosc', 'w+b') as file_blosc:
                file_blosc.write(blosc.compress(dill.dumps(image)))
            if args.delete:
                os.remove(impath)

if __name__ == "__main__":
    sys.exit(compress())
