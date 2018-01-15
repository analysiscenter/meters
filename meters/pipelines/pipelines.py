"""Contains class to configurate pipelines"""
import numpy as np
import sys 

from ..dataset import Pipeline, F, V

_DEFAULT_CONFIG = {
    'n_digits': 8,
    'task': 'classification',
    'optimizer': 'Adam',
    'loss': 'se',
    'metrics': 'accuracy',
    'placeholders_config': None,
    'model': None,
    'model_name': None,
    'fetches': 'accuracy',
    'variables_name': 'accuracy',
    'batch_size': 1,
    'n_epochs': 1
}
class PipelineFactory:

    def __init__(self, static_config=None):
        static_config = static_config or _DEFAULT_CONFIG
        self.update_config(static_config)

    def update_config(self, config):
        for key, value in config.items():
            if key in _DEFAULT_CONFIG.keys():
                _DEFAULT_CONFIG[key] = value

    def simple_load(self, path_to_images=None, path_to_labels=None, path_to_coord=None):
        """Load data from path with default format.
        For images default is blocs format and 'images' as the name of components.
        For labels default is csv format and 'labels' as the name of components.
        For coordinates default is csv format and 'coordinates' as the name of components.

        Parameters
        ----------
        path_to_images : str
            path to folder with images
        path_to_labels : str
            path to file with labels table
        path_to_coord : str
            path to file with coordinates table

        Returns
        -------
            pipeline with load functions"""
        path_to_images = path_to_images or './data/images'
        path_to_labels = path_to_labels or './data/labels/labels.csv'
        path_to_coord = path_to_coord or './data/labels/coord.csv'
        load_ppl = (Pipeline()
                    .load(path_to_images, fmt='blosc', components='images')
                    .load(path_to_labels, fmt='csv', components='labels', index_col='file_name')
                    .load(path_to_coord, fmt='csv', components='coordinates', index_col='file_name'))
        return load_ppl

    def simple_crop_data(self, normalize=True):
        """Cropping data to separate numbers"""
        crop_ppl = (Pipeline()
                    .crop_to_bbox()
                    .crop_to_numbers())
        if normalize:
            crop_ppl += Pipeline().normalize_images()
        return prepare_ppl

    def simple_train(self):
        pass

    def create_predict(self, model_name, pipeline, metrics=None):
        pass    