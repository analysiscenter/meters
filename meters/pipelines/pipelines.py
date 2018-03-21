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
    The default values for the remaining parameters are listed below:

    ``model_type`` : 'dynamic'

    ``mode`` : 'a'

    ``batch_size`` : 1

    ``shuffle``, ``drop_last`` : True

    ``model`` , ``model_name``, ``n_epochs`` : None

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
        'drop_last': True
    }

def default_model_config():
    """Create default model config.

    About parameters you can read in the :func:`model <dataset.models>` documentation.

    The default values are listed below:

    ``inputs`` :

        ``images`` : shape = (64, 32, 3)

        ``labels`` : classes = 10, transform = 'ohe', name = 'targets'

    ``optimizer`` : Adam

    ``loss`` : ce

    ``input_block/inputs`` : 'images'

    ``ouput`` : 'labels', 'proba', 'accuracy'

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
    """Create dict with default names of the components

    The initial values are the same as the names of the components.

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
    """ Сontains methods for quickly creating simple pipelines.

    Parameters
    ----------
    pipeline_config : dict
        pipeline's config
    model_config : dict
        model's config
    model_pipeline : pipeline
        contains all loaded models from anothers pipelines
    available_models : list
        list with names of loaded models
    """
    def __init__(self, pipeline_config=None, model_config=None):
        self.pipeline_config = default_config()
        self.model_config = default_model_config()
        self._update_config(pipeline_config, model_config)
        self.model_pipeline = Pipeline()
        self.available_models = []

    def add_model(self, model_name, model):
        """ Save the model to the pipeline in order to avoid
        re-loading the model when re-creating the pipeline.

        Parameters
        ----------
        model_name : str
            the name of the model
        model : str
            the path to the directory in which the model was saved
        """
        if model_name in self.available_models:
            return
        self.available_models.append(model_name)
        import_model = Pipeline().init_model('static', TFModel, model_name,
                                             config={'load' : {'path' : model},
                                                     'build': False})
        self.model_pipeline += import_model

    def _update_config(self, pipeline_config=None, model_config=None):
        """Get all parameters from ``pipeline_config`` and ``model_config`` and update internal configs

        Parameters
        ----------
        pipeline_config : dict
            pipeline's config
        model_config : dict
            model's config
        """
        self.pipeline_config.update(pipeline_config if pipeline_config is not None else '')
        self.model_config.update(model_config if model_config is not None else '')

    def _update_components(self, components):
        """Replaces standard component names with names from ``components``.

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
        """Writes the data for components to a dictionary of the form:

        key : component's name

        value : data for this component

        Parameters
        ----------
        data : DataFrame
            inputs data
        fmt : str
            data format
        components : list or str
            the names of the components into which the data will be loaded.

        Returns
        -------
        dict with keys - names of the compoents and values - data for these components.
        """
        _ = fmt, args, kwargs
        _comp_dict = dict()

        for comp in components:
            if 'labels' not in comp:
                _comp_dict[comp] = data[data.columns[:-1]].values.astype(int)
            else:
                _comp_dict[comp] = data[data.columns[-1]].values

        return _comp_dict

    def load_all(self, src, components=None):
        """Load data from path with certain format.
        Standard path to images is ``src + '/images'``, for labels and coordinates is ``src + '/labels/data.csv'``

        Format of the images must be ``blosc``. Images will be saved as components['images'] component.

        Labels and coordinates are expected to be loaded from csv file. Labels and coordinates will be saved as
        components['labels'] and components['coordinates'] components.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        components : dict, optional
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

    def make_digits(self, shape=(64, 32), is_training=True, components=None, pipeline_config=None):
        """Сrop images by ``coordinates`` and extract digits from them

        Parameters
        ----------
        shape : tuple or list
            shape of the input images (original images will be resized if their shape is different), default (64, 32)
        is_training : bool, optional
            if ``True`` labels will be loaded, default True
        components : dict, optional
            new names of the components
        pipeline_config : dict, optional
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
        make_ppl += Pipeline().split_to_digits(n_digits=self.pipeline_config['n_digits'], is_training=is_training)
        make_ppl = make_ppl + Pipeline().resize(output_shape=shape, preserve_range=True,
                                                src=components['images']) if shape is not None else make_ppl

        return make_ppl

    def add_lazy_run(self, pipeline):
        """Add lazy run at the end of the pipeline

        Parameters
        ----------
        pipeline
            dataset pipeline

        Returns
        -------
        lazy run pipeline
        """
        if not self.pipeline_config['lazy']:
            return pipeline
        return pipeline.run(self.pipeline_config['batch_size'],
                            n_epochs=self.pipeline_config['n_epochs'],
                            shuffle=self.pipeline_config['shuffle'],
                            drop_last=self.pipeline_config['drop_last'],
                            lazy=True)

    def train_to_digits(self, src, shape=(64, 32), components=None, pipeline_config=None, model_config=None):
        """A pipeline is created to train the model to classify the numbers. If ```config['lazy']``` is True,
        the pipeline with lazy run will be returned.

        Simple train includes:

        * load images, coordinates and labels
        * make digits
        * normalize images
        * init model
        * train model
        * save loss value at each iteration into a pipeline variable named ```current_loss```.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        shape : tuple or list, optional
            shape of the input images (original images will be resized if their shape is different), default (64, 32)
        components : dict, optional
            new names of the components
        pipeline_config : dict, optional
            pipeline's config
        model_config : dict, optional
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
            .init_model(self.pipeline_config['model_type'],
                        self.pipeline_config['model'],
                        self.pipeline_config['model_name'],
                        config=self.model_config)
            .train_model(self.pipeline_config['model_name'],
                         fetches='loss',
                         feed_dict={'images': B(components['images']),
                                    'labels': B(components['labels'])},
                         save_to=V('current_loss'),
                         mode=self.pipeline_config['mode'])
        )
        if self.pipeline_config['save']:
            path = self.pipeline_config['model_name'] if self.pipeline_config['model_path'] \
                                                      is None else self.pipeline_config['model_path']
            train_ppl.save_model(self.pipeline_config['model_name'], path=path)

        return self.add_lazy_run(train_ppl)

    def predict_digits(self, shape=(64, 32), is_training=True, src=None, components=None, pipeline_config=None):
        """"A pipeline is created to predict the digits. If ```config['lazy']``` is True,
        the pipeline with lazy run will be returned.

        Simple predict includes:

        * laod images, coordinates and labels
        * make separate digits
        * normalize images
        * import model
        * predict model
        * save prediction to variable named ```predictions```.

        Parameters
        ----------
        shape : tuple or list, optional
            shape of the input images (original images will be resized if their shape is different), default (64, 32)
        is_training : bool, optional
            if ``True`` labels will be loaded, default True
        src : str, optional
            path to the folder with images and labels
        components : dict, optional
            new names of the components
        pipeline_config : dict, optional
            pipeline's config

        Returns
        -------
        pipeline to model prediction
        """
        self._update_config(pipeline_config)
        components = self._update_components(components)

        model_name = self.pipeline_config['model_name']
        model = self.pipeline_config['model']

        pred_ppl = self.load_all(src) if src is not None else Pipeline()
        pred_ppl += self.make_digits(shape=shape, is_training=is_training, components=components)
        if isinstance(model, str):
            if model_name not in self.available_models:
                self.add_model(model_name, model)
            pred_ppl += Pipeline().import_model(model_name, self.model_pipeline)
        else:
            pred_ppl += Pipeline().import_model(model_name, model)
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
                           mode=self.pipeline_config['mode'])
        )

        return self.add_lazy_run(pred_ppl)

    def train_to_coordinates(self, src, components=None, pipeline_config=None, model_config=None):
        """Create train pipeline to predict the coordinates. If ```config['lazy']``` is True,
        the pipeline with lazy run will be returned.

        Simple train includes:

        * load images
        * load coordinates
        * resize images to (120, 120)
        * normalize images
        * init model
        * train model
        * save loss value at each iteration into a pipeline variable named ```current_loss```.

        Parameters
        ----------
        src : str
            path to the folder with images and labels
        components : dict, optional
            new names of the components
        pipeline_config : dict, optional
            pipeline's config
        model_config : dict, optional
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
                                             .init_model(self.pipeline_config['model_type'],
                                                         self.pipeline_config['model'],
                                                         self.pipeline_config['model_name'],
                                                         config=self.model_config)
                                             .train_model(self.pipeline_config['model_name'],
                                                          fetches='loss',
                                                          feed_dict={'images': B(components['resized_images']),
                                                                     'labels': B(components['coordinates'])},
                                                          save_to=V('current_loss'),
                                                          mode=self.pipeline_config['mode']))

        if self.pipeline_config['save']:
            path = self.pipeline_config['model_name'] if self.pipeline_config['model_path'] \
                                                      is None else self.pipeline_config['model_path']
            train_bb_ppl.save_model(self.pipeline_config['model_name'], path=path)

        return self.add_lazy_run(train_bb_ppl)

    def predict_coordinates(self, src=None, components=None, pipeline_config=None):
        """Create pipeline to predict coordinates. If ```config['lazy']``` is True,
        the pipeline with lazy run will be returned.

        Simple train includes:

        * load images
        * import model
        * resize images
        * normalize images
        * predict model
        * save the model predictions at each iteration.

        Parameters
        ----------
        src : str, optional
            path to the folder with images and labels
        components : dict, optional
            new names of the components
        pipeline_config : dict, optional
            pipeline's config

        Returns
        -------
        pipeline to the model prediction
        """
        self._update_config(pipeline_config)
        components = self._update_components(components)

        model_name = self.pipeline_config['model_name']
        model = self.pipeline_config['model']

        pred_bb_ppl = Pipeline().load(src=src, fmt='image', components=components['images']) if src is not None \
                                                                                                  else Pipeline()
        if isinstance(model, str):
            if model_name not in self.available_models:
                self.add_model(model_name, model)
            pred_bb_ppl += Pipeline().import_model(model_name, self.model_pipeline)
        else:
            pred_bb_ppl += Pipeline().import_model(model_name, model)

        pred_bb_ppl += (
            Pipeline()
            .resize(output_shape=(120, 120), preserve_range=True,
                    src=components['images'],
                    dst=components['resized_images'])
            .multiply(multiplier=1/255., preserve_type=False,
                      src=components['resized_images'],
                      dst=components['resized_images'])
            .init_variable('predictions', init_on_each_run=list)
            .predict_model(model_name,
                           fetches='predictions',
                           feed_dict={'images': B(components['resized_images'])},
                           save_to=B(components['coordinates']),
                           mode='w')
            .get_global_coordinates(src=components['coordinates'], img=components['images'])
            .update_variable('predictions', B(components['coordinates']), mode=self.pipeline_config['mode'])
        )

        return self.add_lazy_run(pred_bb_ppl)

    def full_prediction(self, src, shape=(64, 32), components_coord=None, pipeline_config_coord=None,
                        components_digits=None, pipeline_config_digits=None):
        """Create simple predict pipeline with predicted coordinates of bbox, croped display and
        predict numbers on the display. If ```config['lazy']``` is True, the pipeline with lazy
        run will be returned.

        Simple train includes:

        * load images
        * import model for coordinates prediction
        * model prediction
        * save the model predictions at each iteration into batch component named components['coordinates']
        * import model for digits prediction
        * model prediction
        * save the predicted numbers at each iteration into pipeline variable named ``predictions``.

        Parameters
        ----------
        src : str
            path to the folder with images and labels for first model
        shape : tuple or list, optional
            shape of the input images (original images will be resized if their shape is different), default (64, 32)
        components_coord : dict, optional
            new names of the components for pipeline with coordinates prediction
        pipeline_config_coord : dict, optional
            a configuration dict for pipeline that predicts the coordinates
        components_digits : dict, optional
            new names of the components for pipeline with digits prediction
        pipeline_config_digits : dict, optional
            a configuration dict for pipeline that predicts the numbers.

        Returns
        -------
        pipeline to the model prediction
        """

        pred_num = self.predict_coordinates(src=src, components=components_coord, pipeline_config=pipeline_config_coord)
        pred_num += self.predict_digits(shape=shape, is_training=False,
                                        components=components_digits, pipeline_config=pipeline_config_digits)

        return self.add_lazy_run(pred_num)
