"""Contains class to configurate pipelines"""
import numpy as np

from ..dataset.dataset import Pipeline, V

_CONFIG = {
    'n_digits': 8,
    'model_type': 'dynamic',
    'mode': 'a',
    'task': 'classification',
    'shape': [64, 32, 3],
    'model': None,
    'model_name': None,
    'fetches': 'output_accuracy',
    'batch_size': 25,
    'n_epochs': 1,
    'shuffle': True,
    'drop_last': True
}

_MODEL_CONFIG = {
    'inputs': {'images': {'shape': (64, 32, 3)},
               'labels': {'classes': (10),
                          'transform': 'ohe',
                          'name': 'targets'}},
    'optimizer': 'Adam',
    'loss': 'ce',
    'input_block/inputs': 'images',
    'head/units': 10,
    'output': dict(ops=['labels', 'proba', 'accuracy'])
}

class PipelineFactory:
    """Consist function for easily creating pipelines

    Parameters
    ----------
    static_config : dict
        configuraton for pipeline and model"""
    def __init__(self, static_config=None):
        self.config = _CONFIG.copy()
        self.model_config = _MODEL_CONFIG.copy()
        self.update_config(static_config)

    def update_config(self, config):
        """Get all parameters from ``config`` and update internal config

        Parameters
        ----------
        config : dict
            configuration to pipeline"""
        if config is None:
            return
        for key, value in config.items():
            if key in self.config.keys():
                self.config[key] = value
            elif key in self.model_config.keys():
                self.model_config[key] = value

    def make_separate_digits(self, batch, model):
        """Changes the shape of the data from (batch,) to [batch, image_sahpe]

        Parameters
        ----------
        batch : Dataset batch
            created batch

        model : Dataset model
            model

        Returns
        -------
        res_dict : dict
            feed dict to model """
        _ = model, self
        x = np.concatenate(np.concatenate(batch.digits))
        x = x.reshape(-1, *self.config['shape'])
        y = np.concatenate(batch.labels).reshape(-1)
        res_dict = {'feed_dict': {'images': x, 'labels': y}}
        return res_dict

    def load_all(self, src):
        """Load data from path with default format.
        path to images is ``src`` + /images
        path to labels and coordinates is ``src`` + /labels/labels.csv(coord.csv)
        For images default is blocs format and 'images' as the name of components.
        For labels default is csv format and 'labels' as the name of components.
        For coordinates default is csv format and 'coordinates' as the name of components.

        Parameters
        ----------
        src : str
            path to data

        Returns
        -------
            pipeline with load functions"""
        path_to_images = src + '/images'
        path_to_labels = src + '/labels/labels.csv'
        path_to_coord = src + '/labels/coord.csv'
        load_ppl = (Pipeline()
                    .load(path_to_images, fmt='blosc', components='images')
                    .load(path_to_labels, fmt='csv', components='labels', index_col='file_name')
                    .load(path_to_coord, fmt='csv', components='coordinates', index_col='file_name'))
        return load_ppl

    def crop_digits(self):
        """Cropping data to separate numbers

        Returns
        -------
            pipeline with action to crop data and to separate digits"""
        crop_ppl = (Pipeline()
                    .normalize_images()
                    .crop_to_bbox()
                    .crop_to_digits(shape=self.config['shape'][:2], n_digits=self.config['n_digits'])
                    .crop_labels())
        return crop_ppl

    def add_lazy_run(self, ppl):
        """Create lazy run pipeline

        Parameters
        ----------
        ppl : Dataset pipeline
            pipeline

        Returns
        -------
            Lazy run pipeline"""
        _ = self
        return ppl.run(self.config['batch_size'],
                       n_epochs=self.config['n_epochs'],
                       shuffle=self.config['shuffle'],
                       drop_last=self.config['drop_last'],
                       lazy=True)

    def simple_train(self, src, model_config=None, ppl_config=None):
        """Create default train pipeline with lazy run

        Parameters
        ----------
        src : str
            path to data

        model_config : dict
            config for model

        ppl_config : dict
            config for pipeline

        Returns
        -------
            pipeline witch can train model"""
        self.update_config(ppl_config)
        self.update_config(model_config)

        train_ppl = self.load_all(src) + self.crop_digits()
        train_ppl += (Pipeline()
                      .init_variable('loss', init_on_each_run=list)
                      .init_model(self.config['model_type'],
                                  self.config['model'],
                                  self.config['model_name'],
                                  config=self.model_config)
                      .train_model(self.config['model_name'],
                                   fetches='loss',
                                   make_data=self.make_separate_digits,
                                   save_to=V('loss'),
                                   mode=self.config['mode']))

        return self.add_lazy_run(train_ppl)


    def simple_predict(self, src, model_name, pipeline, config=None):
        """Create default predict pipeline with lazy run

        Parameters
        ----------
        src : str
            path to data

        model_name : str
            name of the model in pipeline

        pipeline : Dataset pipeline
            pipeline

        config : dict
            configuration dict for pipeline

        Returns
        -------
            pipeline witch can train model"""
        self.update_config(config)
        pred_ppl = self.load_all(src) + self.crop_digits()
        pred_ppl += (Pipeline()
                     .init_variable('prediction', init_on_each_run=list)
                     .import_model(model_name, pipeline)
                     .predict_model(model_name,
                                    fetches='targets',
                                    make_data=self.make_separate_digits,
                                    save_to=V('prediction'),
                                    mode=self.config['mode']))

        return self.add_lazy_run(pred_ppl)
