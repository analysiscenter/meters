"""Contains class to configurate pipelines"""
import numpy as np

from ..dataset.dataset import Pipeline, V, B
from ..dataset.dataset.models.tf import TFModel

def default_config():
    """Create default config for pipeline

    Dict configuration:
    ```n_digits``` - (int) the number of digits on the meter. Default 8.
    ```shape``` - (array) the size of the separate digits. Default [64, 32, 3]
    ```bbox_comp``` - (str) the name of the component with coordinates of bboxes. Default 'cooridnates'

    About another parameters read in the pipeline documentation.

    Returns
    -------
    dict with default pipeline configuration
    """
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
        'bbox_comp': 'coordinates',
        'drop_last': True,
        'feed_dict' : None
    }

def default_model_config():
    """Create default model config.

    About parameters you can read in the model documentation.

    Returns
    -------
    dict with default model configuration
    """
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
    ppl_config : dict
            pipeline's config
    model_config : dict
            model's config
    """
    def __init__(self, ppl_config=None, model_config=None):
        self.config = default_config()
        self.model_config = default_model_config()
        self._update_config(ppl_config, model_config)

    def _update_config(self, ppl_config=None, model_config=None):
        """Get all parameters from ``config`` and update internal configs

        Parameters
        ----------
        ppl_config : dict
            pipeline's config
        model_config : dict
            model's config
        """
        self.config.update(ppl_config if ppl_config is not None else '')
        self.model_config.update(model_config if model_config is not None else '')

    def _load_func(self, data, src, fmt, components=None, *args, **kwargs):
        _ = src, fmt, args, kwargs
        _comp_dict = dict()
        for comp in components:
            if 'labels' not in comp:
                _comp_dict[comp] = data[data.columns[:-1]].values
            else:
                _comp_dict[comp] = data[data.columns[-1]].values
        return _comp_dict

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

        load_ppl = (
            Pipeline()
            .load(src=path_to_images, fmt='image', components='images')
            .load(src=path_to_data, fmt='csv', components=['coordinates', 'labels'],
                  call=self._load_func, index_col='file_name')
        )
        return load_ppl

    def make_digits(self, shape=None, src='images', ppl_config=None, model_config=None):
        """Сrop images by ``coordinates`` and extract digits from them

        Parameters
        ----------
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        src : str
            the name of the component with images
        ppl_config : dict
            pipeline's config
        model_config : dict
            model's config
        Returns
        -------
        pipeline with action that makes separate digits from images
        """
        self._update_config(ppl_config, model_config)
        make_ppl = (
            Pipeline()
            .crop_from_bbox(src=src, comp_coord=self.config['bbox_comp'])
            .split_labels()
            .split_to_digits(n_digits=self.config['n_digits'])
        )
        make_ppl = make_ppl + Pipeline().resize(shape) if shape is not None else make_ppl
        make_ppl += Pipeline().multiply(multiplier=1/255., preserve_type=False)
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

    def train_on_pred_bb(self, src, images_src, model_name, model_path,
                         shape, ppl_config=None, model_config=None):
        """ Create simple train pipeline with predicted coordinates and lazy run at the end.

        Simple train includes:

        * load augmented images
        * load normal images
        * import model
        * predict model
        * make digits
        * init model
        * train model
        * save loss value at each iteration.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        images_src : str
            path to the folder with full size images
        model_name : str
            name of the model in pipeline
        model_path
            path to model
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        ppl_config : dict
            pipeline's config
        model_config : dict
            model's config

        Returns
        -------
        pipeline to train the model
        """
        self._update_config(ppl_config, model_config)

        train_ppl = self.load_all(src) + (Pipeline()
                                          .load(src=images_src, fmt='image', components='full_images'))
        train_ppl += (
            Pipeline()
            .init_model(self.config['model_type'], TFModel, model_name,
                        config={'load' : {'path' : model_path},
                                'build': False})
            .predict_model(model_name,
                           fetches='targets',
                           feed_dict={'images': B('images'),
                                      'labels': B('coordinates')},
                           save_to=B('pred_coordinates'),
                           mode='w')
        )

        train_ppl += self.make_digits(src='full_images', shape=shape)

        train_ppl += (
            Pipeline()
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
                         mode=self.config['mode'])
        )

        return self.add_lazy_run(train_ppl)

    def simple_train_bb(self, src, ppl_config=None, model_config=None):
        """Create simple train pipeline to predict the coordinates. At the end of the pipeline is added a lazy run.

        Simple train includes:

        * load augmented images
        * load coordinates
        * init model
        * train model
        * save loss value at each iteration.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        ppl_config : dict
            pipeline's config
        model_config : dict
            model's config
        Returns
        -------
        pipeline to train the model
        """
        self._update_config(ppl_config, model_config)

        train_bb_ppl = self.load_all(src) + (Pipeline()
                                             .init_variable('current_loss', init_on_each_run=list)
                                             .init_model(self.config['model_type'],
                                                         self.config['model'],
                                                         self.config['model_name'],
                                                         config=self.model_config)
                                             .train_model(self.config['model_name'],
                                                          fetches='loss',
                                                          feed_dict={'images': B('images'),
                                                                     'labels': B('coordinates')},
                                                          save_to=V('current_loss'),
                                                          mode=self.config['mode']))

        return self.add_lazy_run(train_bb_ppl)

    def simple_predict_bb(self, src, model_name, pipeline, ppl_config=None):
        """Create simple train pipeline with predicted coordinates and lazy run at the end.

        Simple train includes:

        * load augmented images
        * import model
        * predict model
        * save the model predictions at each iteration.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        model_name : str
            name of the model in pipeline
        pipeline
            Dataset pipeline
        ppl_config : dict
            pipeline's config

        Returns
        -------
        pipeline to the model prediction
        """
        self._update_config(ppl_config)

        pred_bb_ppl = self.load_all(src) + (Pipeline()
                                            .import_model(model_name, pipeline)
                                            .init_variable('prediction', init_on_each_run=list)
                                            .predict_model(model_name,
                                                           fetches='targets',
                                                           feed_dict={'images': B('images'),
                                                                      'labels': B('coordinates')},
                                                           save_to=V('prediction'),
                                                           mode=self.config['mode']))
        return self.add_lazy_run(pred_bb_ppl)

    def simple_train(self, src, shape=(64, 32), ppl_config=None, model_config=None):
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
        ppl_config : dict
            pipeline's config
        model_config : dict
            model's config
        Returns
        -------
        pipeline to train the model
        """
        self._update_config(ppl_config, model_config)
        train_ppl = self.load_all(src) + self.make_digits(shape=shape)
        train_ppl += (
            Pipeline()
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
                         mode=self.config['mode'])
        )

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
        ppl_config : dict
            configuration dict for pipeline

        Returns
        -------
        pipeline to model prediction
        """
        self._update_config(ppl_config)
        pred_ppl = self.load_all(src) + self.make_digits(shape=shape)
        pred_ppl += (
            Pipeline()
            .init_variable('prediction', init_on_each_run=list)
            .import_model(model_name, pipeline)
            .predict_model(model_name,
                           fetches='targets',
                           feed_dict={'images': B('images'),
                                      'labels': B('labels')},
                           save_to=V('prediction'),
                           mode=self.config['mode'])
        )

        return self.add_lazy_run(pred_ppl)
