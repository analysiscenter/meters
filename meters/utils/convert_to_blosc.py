"""File to compless images to blosc format. Can use with aruments from command line.
If path_from and path_to is empty - images will getting from working directory and blosc files saved in the same place.
Arguments
---------
-i : string
    A path or name of directory to images

-o : string
    Name of directory or path to created blosc files

-d : bool
    delete all images
"""
import os
import sys
import argparse

import dill
import blosc
from tqdm import tqdm
import matplotlib.pyplot as plt

def compress():
    """Convert images from any format to blosc format."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, help='name of path with images')
    parser.add_argument('-o', '--output', type=str, help='name of path where blosc files will be created')
    parser.add_argument('-d', '--delete', action='store_true', help='delete images')
    args = parser.parse_args()

    path_from = args.images if args.images else './'
    path_to = args.output if args.output else path_from

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
