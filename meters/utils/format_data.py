#  pylint: disable=no-member
"""Converts labels and data to the format used in the experiment.

Note : It is not possible to convert only coordinates as they depend on the labels format.

Arguments
---------
-l : str
    path to the file with labels, include file name.csv

-c : str
    path to the file with coordinates, include file name.csv

-d : str
    path to the file with labels and coordinates

-s : str
    name of the file with labels and coordinates
"""
import os
import re
import sys
import argparse

import pandas as pd
import numpy as np

def main():
    """Convert labels and coordinates from csv to normal format. Normal is ambiguous format"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels', type=str, help='path to the file with labels')
    parser.add_argument('-c', '--coord', type=str, help='path to the file with coordinates')
    parser.add_argument('-d', '--data', type=str, help='path to the file with all data (labels and coordinates)')
    parser.add_argument('-s', '--save', type=str, help='Required argument. Name of the file where new labels and\
                                                        coordinates are saved (folder is the same as for the file\
                                                        with old labels or data)')
    args = parser.parse_args()

    if args.labels:
        labels = format_labels(args.labels)

    if args.coord:
        if args.labels is None:
            raise ValueError("It is not possible to calculate new coordinates without 'labels'.\
                             Use -l to add a file with labels.")
        data = format_coordinates(args.coord, labels)

    if args.data:
        data = format_data(args.data)

    if args.save:
        if args.save[-4:] != '.csv':
            args.save += '.csv'
        path = args.labels or args.data
        path = os.path.join(os.path.split(path)[0], args.save)
        pd.DataFrame.to_csv(data, path)
    else:
        raise ValueError("Missing required argument '-s'")

def format_labels(src):
    """Convert labels from csv to normat format"""
    try:
        labels = pd.read_csv(src, index_col='file_name', usecols=['file_name', 'counter_value'])
    except ValueError:
        bad_labels = pd.read_csv(src)
        cols = bad_labels.columns
        labels = pd.DataFrame(data=bad_labels[cols[3]].values, index=bad_labels[cols[2]].values,    # pylint: disable=redefined-variable-type
                              columns=['counter_value'])

    labels = labels.loc[[name for name in labels.index.values if 'frame' not in name]]
    labels['counter_value'] = [lab.replace('.', ',') for lab in labels.counter_value.values]
    return labels

def format_coordinates(src_coord, labels):
    """Convert coordinates from csv to normat format"""
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

    data = pd.DataFrame(index=labels.index,
                        columns=['coordinates', 'labels'],
                        data=[[str(coord)[1:-1], label] for coord, label in
                              zip(np.concatenate(np.array(numbers_coord)), labels['counter_value'].values)])

    return data

def format_data(src_data):
    """Convert data from csv to normat format"""
    data = pd.read_csv(src_data)
    cols = data.columns

    numbers_coord = []
    for string in data[cols[1]].values:
        numbers_coord.append(list([int(i) for i in re.sub('\\D+', ' ', string[45:-7]).split(' ')[1:]]))
    numbers_coord = np.array([str(coord)[1:-1] for coord in numbers_coord])

    new_data = pd.DataFrame(index=data[cols[0]],
                            columns=['coordinates', 'labels'],
                            data=list(zip(np.array(numbers_coord), data[cols[-1]].values)))
    return new_data

if __name__ == "__main__":
    sys.exit(main())
