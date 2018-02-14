"""Contains class to configurate pipelines"""
from ..dataset.dataset import Pipeline, V, B
from ..dataset.dataset.models.tf import TFModel

def default_config():
    """Create default config for pipeline

    Dict configuration:

    ``n_digits`` - (int) the number of digits on the meter. Default 8.

    ``model_path`` - (str) path to save the model. Default is a name of the model in
    the current directory.

    ``save`` - (bool) if ``True`` - the model will be saved after training. Default ``False``.

    ``lazy`` - (bool) if ``True`` - adds a lazy run at the end. Default ``True``.

    About another parameters read in the :func:`pipeline <dataset.classes.pipeline>` documentation.

    Returns
    -------
    dict with default pipeline configuration
    """
    return {
        'n_digits': 8,
        'model_type': 'dynamic',
        'mode': 'a',
        'model': None,
        'model_name': None,
        'model_path': None,
        'save': False,
        'lazy': True,
        'batch_size': 1,
        'n_epochs': None,
        'shuffle': True,
        'drop_last': True,
        'feed_dict' : None
    }

def default_model_config():
    """Create default model config.

    About parameters you can read in the :func:`model <dataset.models>` documentation.

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
def default_components():
    """Create dict with default names of components

    Returns
    -------
    dict with default components names"""
    return {
        'images': 'images',
        'resized_images': 'resized_images',
        'coordinates': 'coordinates',
        'labels': 'labels'
    }
class PipelineFactory:
    """Consists of methods for pipelines creation

    Parameters
    ----------
    pipeline_config : dict
            pipeline's config
    model_config : dict
            model's config
    """
    def __init__(self, pipeline_config=None, model_config=None):
        self.config = default_config()
        self.model_config = default_model_config()
        self._update_config(pipeline_config, model_config)

    def _update_config(self, pipeline_config=None, model_config=None):
        """Get all parameters from ``config`` and update internal configs

        Parameters
        ----------
        pipeline_config : dict
            pipeline's config
        model_config : dict
            model's config
        """
        self.config.update(pipeline_config if pipeline_config is not None else '')
        self.model_config.update(model_config if model_config is not None else '')

    def _update_components(self, components):
        """Get names from components and update default components dict with new values.

        Parameters
        ----------
        components : dict
            new names of the components

        Returns
        -------
        dict with names of the components
        """
        default_comp = default_components()
        default_comp.update(components if components is not None else '')
        return default_comp

    def _load_func(self, data, _, fmt, components=None, *args, **kwargs):
        """Compares downloaded data with components.

        Parameters
        ----------
        data : DataFrame
            inputs data
        components : list or str
            the names of the components in which to download the data

        Returns
        -------
        dict with keys - names of the compoents and values - data for these components.
        """
        _ = fmt, args, kwargs
        _comp_dict = dict()

        for comp in components:
            if 'labels' not in comp:
                _comp_dict[comp] = data[data.columns[:-1]].values
            else:
                _comp_dict[comp] = data[data.columns[-1]].values

        return _comp_dict

    def load_all(self, src, components=None):
        """Load data from path with certain format.
        Standard path to images is ``src + '/images'``, for labels and coordinates is ``src + '/labels/data.csv'``

        Format of the images must be `blosc`. Images will be saved as 'images' component.

        Labels and coordinates are expected to be loaded from csv file. Labels and coordinates will be saved as
        'labels' and 'coordinates' components.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        components : dict
            new names of the components

        Returns
        -------
        pipeline to load labels, coordinates and images.
        """
        components = self._update_components(components)

        path_to_images = src + '/images'
        path_to_data = src + '/labels/data.csv'
        load_ppl = (
            Pipeline()
            .load(src=path_to_images, fmt='image', components=components['images'])
            .load(src=path_to_data, fmt='csv', components=[components['coordinates'], components['labels']],
                  call=self._load_func, index_col='file_name')
        )

        return load_ppl

    def make_digits(self, shape=None, is_training=True, components=None, pipeline_config=None):
        """Сrop images by ``coordinates`` and extract digits from them

        Parameters
        ----------
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        is_training : bool, optional, default True
            if ``True`` labels will be loaded
        components : dict
            new names of the components
        pipeline_config : dict
            pipeline's config

        Returns
        -------
        pipeline with action that makes separate digits from images
        """
        self._update_config(pipeline_config)
        components = self._update_components(components)

        make_ppl = Pipeline().crop_from_bbox(src=components['images'], component_coord=components['coordinates'])
        if is_training:
            make_ppl += Pipeline().split_labels()
        make_ppl += Pipeline().split_to_digits(n_digits=self.config['n_digits'], is_training=is_training)
        make_ppl = make_ppl + Pipeline().resize(output_shape=shape, preserve_range=True,
                                                src=components['images']) if shape is not None else make_ppl

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
        if not self.config['lazy']:
            return ppl
        return ppl.run(self.config['batch_size'],
                       n_epochs=self.config['n_epochs'],
                       shuffle=self.config['shuffle'],
                       drop_last=self.config['drop_last'],
                       lazy=True)

    def train_to_digits(self, src, shape=(64, 32), components=None, pipeline_config=None, model_config=None):
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
        components : dict
            new names of the components
        pipeline_config : dict
            pipeline's config
        model_config : dict
            model's config

        Returns
        -------
        pipeline to train the model
        """
        self._update_config(pipeline_config, model_config)
        components = self._update_components(components)

        train_ppl = self.load_all(src) + self.make_digits(shape=shape, components=components)
        train_ppl += (
            Pipeline()
            .init_variable('current_loss', init_on_each_run=list)
            .multiply(multiplier=1/255., preserve_type=False,
                      src=components['images'],
                      dst=components['images'])
            .init_model(self.config['model_type'],
                        self.config['model'],
                        self.config['model_name'],
                        config=self.model_config)
            .train_model(self.config['model_name'],
                         fetches='loss',
                         feed_dict={'images': B(components['images']),
                                    'labels': B(components['labels'])},
                         save_to=V('current_loss'),
                         mode=self.config['mode'])
        )
        if self.config['save']:
            path = self.config['model_name'] if self.config['model_path'] is None else self.config['model_path']
            train_ppl.save_model(self.config['model_name'], path=path)

        return self.add_lazy_run(train_ppl)

    def predict_digits(self, shape=(64, 32), is_training=True, src=None, components=None, pipeline_config=None):
        """"A prediction pipeline is created to predict digits. If ```config['lazy']``` is True,
        the pipeline with lazy run will be returned.

        Simple predict includes:

        * laod + make
        * import model
        * predict model
        * save prediction to variable named ```prediction```.

        Parameters
        ----------
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        is_training : bool, optional, default True
            if ``True`` labels will be loaded
        src : str
            path to the folder with images and labels
        components : dict
            new names of the components
        pipeline_config : dict
            pipeline's config

        Returns
        -------
        pipeline to model prediction
        """
        self._update_config(pipeline_config)
        components = self._update_components(components)

        model_name = self.config['model_name']
        model = self.config['model']

        pred_ppl = self.load_all(src) if src is not None else Pipeline()
        pred_ppl += self.make_digits(shape=shape, is_training=is_training, components=components)

        if isinstance(model, str):
            import_model = Pipeline().init_model(self.config['model_type'], TFModel, model_name,
                                                 config={'load' : {'path' : model},
                                                         'build': False})
        else:
            import_model = Pipeline().import_model(model_name, model)

        pred_ppl += import_model
        pred_ppl += (
            Pipeline()
            .multiply(multiplier=1/255., preserve_type=False,
                      src=components['images'],
                      dst=components['images'])
            .init_variable('predictions', init_on_each_run=list)
            .predict_model(model_name,
                           fetches='predictions',
                           feed_dict={'images': B(components['images'])},
                           save_to=V('predictions'),
                           mode=self.config['mode'])
        )

        return self.add_lazy_run(pred_ppl)

    def train_to_coordinates(self, src, components=None, pipeline_config=None, model_config=None):
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
        components : dict
            new names of the components
        pipeline_config : dict
            pipeline's config
        model_config : dict
            model's config
        Returns
        -------
        pipeline to train the model
        """
        self._update_config(pipeline_config, model_config)
        components = self._update_components(components)

        train_bb_ppl = self.load_all(src) + (Pipeline()
                                             .resize(output_shape=(120, 120), preserve_range=True,
                                                     src=components['images'],
                                                     dst=components['resized_images'])
                                             .multiply(multiplier=1/255., preserve_type=False,
                                                       src=components['resized_images'],
                                                       dst=components['resized_images'])
                                             .init_variable('current_loss', init_on_each_run=list)
                                             .init_model(self.config['model_type'],
                                                         self.config['model'],
                                                         self.config['model_name'],
                                                         config=self.model_config)
                                             .train_model(self.config['model_name'],
                                                          fetches='loss',
                                                          feed_dict={'images': B(components['resized_images']),
                                                                     'labels': B(components['coordinates'])},
                                                          save_to=V('current_loss'),
                                                          mode=self.config['mode']))

        if self.config['save']:
            path = self.config['model_name'] if self.config['model_path'] is None else self.config['model_path']
            train_bb_ppl.save_model(self.config['model_name'], path=path)

        return self.add_lazy_run(train_bb_ppl)

    def predict_coordinates(self, src=None, components=None, pipeline_config=None):
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
        components : dict
            new names of the components
        pipeline_config : dict
            pipeline's config

        Returns
        -------
        pipeline to the model prediction
        """
        self._update_config(pipeline_config)
        components = self._update_components(components)

        model_name = self.config['model_name']
        model = self.config['model']

        pred_bb_ppl = Pipeline().load(src=src, fmt='image', components=components['images']) if src is not None \
                                                                                                  else Pipeline()

        if isinstance(model, str):
            import_model = Pipeline().init_model(self.config['model_type'], TFModel, model_name,
                                                 config={'load' : {'path' : model},
                                                         'build': False})
        else:
            import_model = Pipeline().import_model(model_name, model)

        pred_bb_ppl += import_model
        pred_bb_ppl += (
            Pipeline()
            .resize(output_shape=(120, 120), preserve_range=True,
                    src=components['images'],
                    dst=components['resized_images'])
            .multiply(multiplier=1/255., preserve_type=False,
                      src=components['resized_images'],
                      dst=components['resized_images'])
            .predict_model(model_name,
                           fetches='predictions',
                           feed_dict={'images': B(components['resized_images'])},
                           save_to=B(components['coordinates']),
                           mode='w')
            .get_global_coordinates(src=components['coordinates'], img=components['images'])
        )

        return self.add_lazy_run(pred_bb_ppl)

    def full_prediction(self, src, shape=(64, 32), components_coord=None, pipeline_config_coord=None,
                        components_digits=None, pipeline_config_digits=None):
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
        src : str
            path to the folder with images and labels for first model
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different)
        components_coord : dict
            new names of the components for pipeline with coordinates prediction
        pipeline_config_coord : dict
            a configuration dict for pipeline that predicts the coordinates
        components_digits : dict
            new names of the components for pipeline with digits prediction
        pipeline_config_digits : dict
            a configuration dict for pipeline that predicts the numbers.

        Returns
        -------
        pipeline to the model prediction
        """

        pred_num = self.predict_coordinates(src=src, components=components_coord, pipeline_config=pipeline_config_coord)
        pred_num += self.predict_digits(shape=shape, is_training=False,
                                        components=components_digits, pipeline_config=pipeline_config_digits)

        return self.add_lazy_run(pred_num)
