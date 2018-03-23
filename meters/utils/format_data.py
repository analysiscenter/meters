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
    """Convert labels and coordinates from csv the format used in the experiment"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels', type=str, help='path to the file with labels')
    parser.add_argument('-c', '--coord', type=str, help='path to the file with coordinates')
    parser.add_argument('-d', '--data', type=str, help='path to the file with all data (labels and coordinates)')
    parser.add_argument('-s', '--save', type=str, help='Required argument. Name of the file where new labels and\
                                                        coordinates are saved (folder is the same as for the file\
                                                        with old labels or data)', required=True)
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

        folder_path = [os.path.join(folder_name) for folder_name in os.path.split(path)[:-1]][0]
        folder_path = '.' if folder_path == '' else folder_path

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pd.DataFrame.to_csv(data, path)
    else:
        raise ValueError("Missing required argument '-s'")


def format_labels(src):
    """Convert labels from csv to the format used in the experiment."""

    try:
        labels = pd.read_csv(src, index_col='file_name', usecols=['file_name', 'counter_value'])
    except ValueError:
        bad_labels = pd.read_csv(src)
        cols = bad_labels.columns
        labels = pd.DataFrame(data=bad_labels[cols[-1]].values, index=bad_labels[cols[-2]].values,    # pylint: disable=redefined-variable-type
                              columns=['counter_value'])
    labels = labels.loc[[name for name in labels.index.values if 'frame' not in name]]
    labels['counter_value'] = [lab.replace('.', ',') for lab in labels.counter_value.values]
    return labels

def format_coordinates(src_coord, labels):
    """Convert coordinates from csv to the format used in the experiment."""
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
                        columns=['x', 'y', 'h', 'w', 'labels'],
                        data=[[*coord, label] for coord, label in
                              zip(np.concatenate(numbers_coord), labels['counter_value'].values)])

    return data

def format_data(src_data):
    """Convert data from csv to the format used in the experiment."""
    try:
        data = pd.read_csv(src_data)
    except: # pylint: disable=bare-except
        data = pd.read_excel(src_data)  # pylint: disable=redefined-variable-type

    cols = data.columns
    names = ['x', 'y', 'width', 'height']
    values = []
    bad_vals = []
    coordinates = dict(zip(names, [0]*4))

    for j, string in enumerate(data[cols[1]].values):
        ix_coord = sorted({coord : string.find(coord) for coord in coordinates.keys()}.items(), key=lambda x: x[1])

        for i, idx in enumerate(ix_coord):
            if i+1 < len(ix_coord):
                coordinates[idx[0]] = (re.sub('[^0-9.]', '', string[idx[1]: ix_coord[i+1][1]]))
            else:
                coordinates[idx[0]] = (re.sub('[^0-9.]', '', string[idx[1]: ]))

        if '' in coordinates.values():
            bad_vals.append(j)
            continue

        values.append(list(map(lambda name: int(np.float32(coordinates[name])), names)))

    data = data.drop(bad_vals)
    labels = list(map(lambda a: str(a).replace('.', '.'), data[cols[-1]].values))
    data.columns = ['file_name'] + list(data.columns[1:])
    data['file_name'] = ['a' + '0' * (5 - len(str(d))) + str(d) for d in data[data.columns[0]].values]
    new_data = pd.DataFrame(index=data['file_name'],
                            columns=['x', 'y', 'h', 'w', 'labels'],
                            data=[[*coord, label] for coord, label in
                                  list(zip(np.array(values), labels))])
    return new_data

if __name__ == "__main__":
    sys.exit(main())
