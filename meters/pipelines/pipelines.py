"""Contains class to configurate pipelines"""
import numpy as np

from ..dataset.dataset import Pipeline, V, B


def default_config():
    """Create default config for pipeline"""
    return {
        'n_digits': 8,
        'model_type': 'dynamic',
        'mode': 'a',
        'task': 'classification',
        'shape': np.array([64, 32, 3]),
        'model': None,
        'model_name': None,
        'fetches': 'output_accuracy',
        'batch_size': 1,
        'n_epochs': None,
        'shuffle': True,
        'drop_last': True
    }

def default_model_config():
    """Create default model config"""
    return {
        'inputs': {'images': {'shape': (64, 32, 3)},
                   'labels': {'classes': (10),
                              'transform': 'ohe',
                              'name': 'targets'}},
        'optimizer': 'Adam',
        'loss': 'ce',
        'input_block/inputs': 'images',
        'output': dict(ops=['labels', 'proba', 'accuracy'])
    }

class PipelineFactory:
    """Consists of methods for pipelines creation

    Parameters
    ----------
    config : dict
        configuraton of the pipeline

    model_config : dict
        configuraton of the model
    """
    def __init__(self, ppl_config=None, model_config=None):
        self.config = default_config()
        self.model_config = default_model_config()
        self._update_config(ppl_config, model_config)

    def _update_config(self, ppl_config=None, model_config=None):
        """Get all parameters from ``config`` and update internal configs

        Parameters
        ----------
        config : dict
            configuration of a pipeline
        """
        self.config.update(ppl_config if ppl_config is not None else '')
        self.model_config.update(model_config if model_config is not None else '')

    def load_all(self, src):
        """Load data from path with certain format.
        Standard path to images is ``src + '/images'``, for labels and coordinates is ``src + '/labels/data.csv'``

        Format of the images must be `blosc`. Images will be saved as 'images' component.

        Labels and coordinates are expected to be loaded from csv file. Labels and coordinates will be saved as
        'labels' and 'coordinates' components.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        """
        path_to_images = src + '/images'
        path_to_data = src + '/labels/data.csv'
        load_ppl = (Pipeline()
                    .load(path_to_images, fmt='blosc', components='images')
                    .load(path_to_data, fmt='csv', components=['coordinates', 'labels'], index_col='file_name'))
        return load_ppl

    def make_digits(self, shape=None):
        """Сrop images by ``coordinates`` and extract digits from them

        Parameters
        ----------
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)

        Returns
        -------
        pipeline with action that makes separate digits from images
        """
        make_ppl = (Pipeline()
                    .crop_from_bbox()
                    .split_labels()
                    .split_to_digits(n_digits=self.config['n_digits']))
        make_ppl = make_ppl + Pipeline().resize(shape=shape) if shape is not None else make_ppl
        make_ppl += Pipeline().normalize_images()
        return make_ppl

    def add_lazy_run(self, ppl):
        """Create lazy run pipeline

        Parameters
        ----------
        ppl
            dataset pipeline

        Returns
        -------
        lazy run pipeline
        """
        return ppl.run(self.config['batch_size'],
                       n_epochs=self.config['n_epochs'],
                       shuffle=self.config['shuffle'],
                       drop_last=self.config['drop_last'],
                       lazy=True)

    def simple_train(self, src, shape=(64, 32), model_config=None, ppl_config=None):
        """Create simple train pipeline with lazy run at the end.

        Simple train includes:

        * load + make
        * init model
        * train model
        * save loss value at each iteration.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        model_config : dict
            model's config
        ppl_config : dict
            config for pipeline

        Returns
        -------
        pipeline to train model
        """
        self._update_config(ppl_config, model_config)
        train_ppl = self.load_all(src) + self.make_digits(shape=shape)
        train_ppl += (Pipeline()
                      .init_variable('loss', init_on_each_run=list)
                      .init_model(self.config['model_type'],
                                  self.config['model'],
                                  self.config['model_name'],
                                  config=self.model_config)
                      .train_model(self.config['model_name'],
                                   fetches='loss',
                                   feed_dict={'images': B('images'),
                                              'labels': B('labels')},
                                   save_to=V('loss'),
                                   mode=self.config['mode']))

        return self.add_lazy_run(train_ppl)


    def simple_predict(self, src, model_name, pipeline, shape=(64, 32), ppl_config=None):
        """Create simple predict pipeline with lazy run at the end.

        Simple predict includes:

        * laod + make
        * import model
        * predict model
        * save prediction to variable named ``prediction``.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        model_name : str
            name of the model in pipeline
        pipeline
            Dataset pipeline
        config : dict
            configuration dict for pipeline

        Returns
        -------
        pipeline to predict model
        """
        self._update_config(ppl_config)
        pred_ppl = self.load_all(src) + self.make_digits(shape=shape)
        pred_ppl += (Pipeline()
                     .init_variable('prediction', init_on_each_run=list)
                     .import_model(model_name, pipeline)
                     .predict_model(model_name,
                                    fetches='targets',
                                    feed_dict={'images': B('images'),
                                               'labels': B('labels')},
                                    save_to=V('prediction'),
                                    mode=self.config['mode']))

        return self.add_lazy_run(pred_ppl)
