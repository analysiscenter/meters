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
        'shape': np.array([64, 32, 3]),
        'model': None,
        'model_name': None,
        'save': False,
        'lazy': True,
        'model_path': None,
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

    def make_digits(self, shape=None, src='images', is_pred=False, ppl_config=None, model_config=None):
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
        make_ppl = Pipeline().crop_from_bbox(src=src, comp_coord=self.config['bbox_comp'])
        if not is_pred:
            make_ppl += Pipeline().split_labels()
        make_ppl += Pipeline().split_to_digits(n_digits=self.config['n_digits'], is_pred=is_pred)
        make_ppl = make_ppl + Pipeline().resize(shape) if shape is not None else make_ppl
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

    def train_to_digits(self, src, shape=(64, 32), ppl_config=None, model_config=None):
        """A training pipeline is created to train model predict digits. If ```config['lazy']``` is True,
        the pipeline with lazy run will be returned.

        Simple train includes:

        * load + make
        * init model
        * train model
        * save loss value at each iteration into a pipeline variable named ```current_loss```.

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
            .init_variable('current_loss', init_on_each_run=list)
            .multiply(multiplier=1/255., preserve_type=False, src='images')
            .init_model(self.config['model_type'],
                        self.config['model'],
                        self.config['model_name'],
                        config=self.model_config)
            .train_model(self.config['model_name'],
                         fetches='loss',
                         feed_dict={'images': B('images'),
                                    'labels': B('labels')},
                         save_to=V('current_loss'),
                         mode=self.config['mode'])
        )
        if self.config['save']:
            path = self.config['model_name'] if self.config['model_path'] is None else self.config['model_path']
            train_ppl.save_model(self.config['model_name'], path=path)

        if self.config['lazy']:
            return self.add_lazy_run(train_ppl)
        return train_ppl

    def predict_digits(self, shape=(64, 32), make_src='images', is_pred=False, src=None, ppl_config=None):
        """"A prediction pipeline is created to predict digits. If ```config['lazy']``` is True,
        the pipeline with lazy run will be returned.

        Simple predict includes:

        * laod + make
        * import model
        * predict model
        * save prediction to variable named ```prediction```.

        Parameters
        ----------
        model_name : str
            name of the model in pipeline
        model : Dataset pipeline, str
            pipeline with trained model or src with saved model
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        src : str
            path to the folder with images and labels
        ppl_config : dict
            pipeline's config

        Returns
        -------
        pipeline to model prediction
        """
        self._update_config(ppl_config)
        model_name = self.config['model_name']
        model = self.config['model']

        pred_ppl = self.load_all(src) if src is not None else Pipeline()
        pred_ppl += self.make_digits(src=make_src, shape=shape, is_pred=is_pred)

        if isinstance(model, str):
            import_model = Pipeline().init_model(self.config['model_type'], TFModel, model_name,
                                                 config={'load' : {'path' : model},
                                                         'build': False})
        else:
            import_model = Pipeline().import_model(model_name, model)

        pred_ppl += import_model
        pred_ppl += (
            Pipeline()
            .multiply(multiplier=1/255., preserve_type=False, src='images')
            .init_variable('digits_prediction', init_on_each_run=list)
            .predict_model(model_name,
                           fetches='predictions',
                           feed_dict={'images': B('images')},
                           save_to=V('digits_prediction'),
                           mode=self.config['mode'])
        )

        if self.config['lazy']:
            return self.add_lazy_run(pred_ppl)
        return pred_ppl

    def train_to_coordinates(self, src, ppl_config=None, model_config=None):
        """Create simple train pipeline to predict the coordinates. At the end of the pipeline is added a lazy run.

        Simple train includes:

        * load augmented images
        * load coordinates
        * init model
        * train model
        * save loss value at each iteration.

        Parameters
        ----------
        src : simple_train
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
                                             .resize(output_shape=(120, 120), preserve_range=False,
                                                     src='images', dst='croped_images')
                                             .multiply(multiplier=1/255., preserve_type=False, src='croped_images')
                                             .init_variable('current_loss', init_on_each_run=list)
                                             .init_model(self.config['model_type'],
                                                         self.config['model'],
                                                         self.config['model_name'],
                                                         config=self.model_config)
                                             .train_model(self.config['model_name'],
                                                          fetches='loss',
                                                          feed_dict={'images': B('croped_images'),
                                                                     'labels': B('coordinates')},
                                                          save_to=V('current_loss'),
                                                          mode=self.config['mode']))
        if self.config['save']:
            path = self.config['model_name'] if self.config['model_path'] is None else self.config['model_path']
            train_bb_ppl.save_model(self.config['model_name'], path=path)

        if self.config['lazy']:
            return self.add_lazy_run(train_bb_ppl)
        return train_bb_ppl

    def predict_coordinates(self, src=None, ppl_config=None):
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
        model : Dataset pipeline, str
            pipeline with trained model or src with saved model
        ppl_config : dict
            pipeline's config

        Returns
        -------
        pipeline to the model prediction
        """
        self._update_config(ppl_config)
        model_name = self.config['model_name']
        model = self.config['model']
        pred_bb_ppl = Pipeline().load(src=src, fmt='image', components='images') if src is not None else Pipeline()

        if isinstance(model, str):
            import_model = Pipeline().init_model(self.config['model_type'], TFModel, model_name,
                                                 config={'load' : {'path' : model},
                                                         'build': False})
        else:
            import_model = Pipeline().import_model(model_name, model)

        pred_bb_ppl += import_model
        pred_bb_ppl += (
            Pipeline()
            .resize(output_shape=(120, 120), preserve_range=False,
                    src='images', dst='croped_images')
            .multiply(multiplier=1/255., preserve_type=False, src='croped_images')
            .init_variable('prediction', init_on_each_run=list)
            .predict_model(model_name,
                           fetches='predictions',
                           feed_dict={'images': B('croped_images')},
                           save_to=B('pred_coordinates'),
                           mode='w')
            .get_global_coordinates(src='pred_coordinates')
            .update_variable('prediction', B('pred_coordinates'), mode=self.config['mode'])
        )

        if self.config['lazy']:
            return self.add_lazy_run(pred_bb_ppl)
        return pred_bb_ppl

    def simple_predict_numbers(self, name_bb_model, bb_model, name_digits_model, digits_model, # pylint: disable=too-many-arguments
                               src, shape=(64, 32), ppl_config_bb=None, ppl_config_digits=None):
        """Create simple predict pipeline with predicted coordinates of bbox, croped display and
        predict numbers on the display.

        Simple train includes:

        * load augmented images
        * import model for coordinates prediction
        * model prediction
        * save the model predictions at each iteration
        * import model for digits prediction
        * model prediction
        * save the predicted numbers at each iteration.

        Parameters
        ----------
        name_bb_model : str
            the name of the model for coordinates prediction
        bb_model : Dataset pipeline, str
            pipeline with trained model for coordinates prediction or src with saved model
        name_digits_model : str
            the name of the model for digits prediction
        digits_model : Dataset pipeline, str
            pipeline with trained model for digits prediction or src with saved model
        src : str
            path to the folder with images and labels for first model
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        ppl_config_bb : dict
            a configuration dict for pipeline that predicts the coordinates
        ppl_config_digits : dict
            a configuration dict for pipeline that predicts the numbers.

        Returns
        -------
        pipeline to the model prediction
        """

        ppl_config_digits['bbox_comp'] = V('prediction')
        ppl_config_bb['model_name'] = name_bb_model
        ppl_config_bb['model'] = bb_model

        ppl_config_digits['model_name'] = name_digits_model
        ppl_config_digits['model'] = digits_model
        pred_num = self.predict_coordinates(src=src, ppl_config=ppl_config_bb)
        pred_num += self.predict_digits(make_src='images', shape=shape, is_pred=True, ppl_config=ppl_config_digits)
        if self.config['lazy']:
            return self.add_lazy_run(pred_num)
        return pred_num
