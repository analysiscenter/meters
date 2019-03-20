"""Contains a function that compresses images into ``blosc`` format. Can be used with aruments from command line.
If path_from and path_to are empty then images are read from the working directory and blosc files are saved in the
same place.

Arguments
---------
-i : string
    Path to the images' directory

-o : string
    Path to the blosc files' directory

-d : bool
    Whether to delete all images

-c : string
    Name of the component to save
"""
import os
import sys
import argparse

import dill
import blosc
from tqdm import tqdm
import matplotlib.pyplot as plt

def compress():
    """Convert images from format which can be loaded by matplotlib to blosc format."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, help="images' folder path")
    parser.add_argument('-o', '--output', type=str, help='name of path where blosc files will be created')
    parser.add_argument('-d', '--delete', action='store_true', help='delete images')
    parser.add_argument('-c', '--component', type=str, help='optional, default=images. Name of the component to save')
    args = parser.parse_args()

    path_from = args.images if args.images else './'
    path_to = args.output if args.output else path_from

    component = args.component if args.component else tuple(['images'],)
    component = tuple([component],) if isinstance(component, str) else component

    print('path from: %s'%path_from)
    print('path to: %s\n'%path_to)

    if not os.path.exists(path_to):
        os.makedirs(path_to)

    files_name = os.listdir(path_from)
    for imfile in tqdm(files_name):
        impath = os.path.join(path_from, imfile)
        if not os.path.isdir(impath):
            image = plt.imread(impath)
            image_path = os.path.join(path_to, 'a' + imfile[:imfile.rfind('.')])
            with open(image_path + '.blosc', 'w+b') as file_blosc:
                data = dict(zip(component, (image,)))
                file_blosc.write(blosc.compress(dill.dumps(data)))
            if args.delete:
                os.remove(impath)

if __name__ == "__main__":
    sys.exit(compress())
