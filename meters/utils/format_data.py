#  pylint: disable=no-member
"""File leading to one type of writing data in labels and coordinates files.

Note : You can't leading only coordinates, because it denended from right labels format.

Arguments
---------
-l : str
    path to file with labels, include file name.csv

-ln : str
    name of file with labels in new format

-c : str
    path to file with coordinates, include file name.csv

-nc : str
    name of file with coordinates in new format
"""
import os
import re
import sys
import argparse

import pandas as pd
import numpy as np

def format_data():
    """Prepare labels in csv format to normal format"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels', type=str, help='path to file with labels')
    parser.add_argument('-nl', '--new_labels', type=str, help='file name labels in new format, \
                                                               which will be save in the save path as old labels')
    parser.add_argument('-c', '--coord', type=str, help='path to file with coordinates')
    parser.add_argument('-nc', '--new_coord', type=str, help='file name with new coordinates, \
                                                              which will be save in the save path as old coordinates')
    args = parser.parse_args()

    if args.labels:
        _format_labels(args.labels, args.new_labels)

    if args.coord:
        labels_path = _create_new_path(args.labels, args.new_labels)
        _format_coordinates(args.coord, labels_path, args.new_coord)

def _format_labels(src, new_name):
    try:
        labels = pd.read_csv(src, index_col='file_name', usecols=['file_name', 'counter_value'])
    except ValueError:
        bad_labels = pd.read_csv(src)
        cols = bad_labels.columns
        labels = pd.DataFrame(data=bad_labels[cols[3]].values, index=bad_labels[cols[2]].values,
                              columns=['counter_value'])
    labels = labels.loc[[name for name in labels.index.values if 'frame' not in name]]
    labels['counter_value'] = [lab.replace('.', ',') for lab in labels.counter_value.values]

    path = _create_new_path(src, new_name)
    pd.DataFrame.to_csv(labels, path)

def _format_coordinates(src_coord, src_labels, new_name):
    labels = pd.read_csv(src_labels)
    try:
        coord = pd.read_csv(src_coord, usecols=['numbers', 'markup']).dropna().reset_index()
    except ValueError:
        bad_coord = pd.read_csv(src_coord)
        cols = bad_coord.columns
        coord = pd.DataFrame(data=bad_coord[[cols[1], cols[7]]].values,
                             columns=['numbers', 'markup']).dropna().reset_index()

    numbers_coord = []
    for i, string in enumerate(coord['markup']):
        list_coord = list([int(i) for i in re.sub('\\D+', ' ', string[36:-7]).split(' ')[1:]])
        numbers_coord.append([list_coord] * int(coord['numbers'].loc[i]))
    new_coord = pd.DataFrame(index=labels.index,
                             columns=['x', 'y', 'width', 'height'],
                             data=np.concatenate(np.array(numbers_coord)))

    path = _create_new_path(src_coord, new_name)
    pd.DataFrame.to_csv(new_coord, path)

def _create_new_path(src, name):
    if name[-4:] != '.csv':
        name += '.csv'
    path = os.path.join(os.path.split(src)[0], name)
    return path

if __name__ == "__main__":
    sys.exit(format_data())
