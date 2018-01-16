"""Batch class for water meter task"""
import scipy
import numpy as np

from ..dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed

class MeterBatch(ImagesBatch):
    """Class to create batch with water meter"""
    components = 'images', 'labels', 'coordinates', 'indices'

    @action
    @inbatch_parallel(init='indices', src='images', post='assemble')
    def normalize_images(self, ind, src='images'):
        """ Normalize pixel values from (0, 255) to (0, 1).

        Parameters
        ----------
        ind : str or int
            dataset index

        src : str
            the name of the placeholder with data

        Retruns
        -------
            normalized images"""
        image = self.get(ind, src)
        normalize_image = image / 255.
        return normalize_image

    def _init_component(self, *args, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices

        Returns
        -------
            array with indices from batch"""
        _ = args, kwargs
        dst = kwargs.get('dst')
        if dst is None:
            raise KeyError('dst argument must be specified')
        if not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self.index)))
        return self.indices

    @action
    @inbatch_parallel(init='_init_component', src='images', dst='bbox', target='threads')
    def crop_to_bbox(self, ind, *args, src='images', dst='bbox', **kwargs):
        """Create cropped attr with crop image use ``coordinates``

        Parameters
        ----------
        ind : str or int
            dataset index

        src : str
            the name of the placeholder with data

        dst : str
            the name of the placeholder in witch the result will be recorded"""
        _ = args, kwargs
        image = self.get(ind, src)
        coord_str = self.get(ind, 'coordinates')
        x, y, width, height = [int(val) for val in coord_str.split()]
        i = self.get_pos(None, src, ind)
        dst_data = image[y:y+height, x:x+width]
        getattr(self, dst)[i] = dst_data

    @action
    @inbatch_parallel(init='_init_component', src='bbox', dst='digits', target='threads')
    def crop_to_digits(self, ind, *args, shape=(64, 32), n_digits=8, src='bbox', dst='digits', **kwargs):
        """Crop image with ``n_digits`` number to ``n_digits`` images with one number

        Parameters
        ----------
        ind : str or int
            dataset index

        shape : tuple or list
            shape of output image

        src : str
            the name of the placeholder with data

        dst : str
            the name of the placeholder in witch the result will be recorded

        n_digits : int
            number of digits on meter"""
        def _resize(img, shape):
            factor = 1. * np.asarray([*shape]) / np.asarray(img.shape[:2])
            if len(img.shape) > 2:
                factor = np.concatenate((factor, [1.] * len(img.shape[2:])))
            new_image = scipy.ndimage.interpolation.zoom(img, factor, order=3)
            return new_image

        _ = args, kwargs
        i = self.get_pos(None, src, ind)
        image = getattr(self, src)[i]
        numbers = np.array([_resize(img, shape) for img in np.array_split(image, n_digits, axis=1)] + [None])[:-1]

        getattr(self, dst)[i] = numbers

    @action
    @inbatch_parallel(init='_init_component', src='labels', dst='labels', target='threads')
    def crop_labels(self, ind, *args, src='labels', dst='labels', **kwargs):
        """Cropped labels from strig to list with separate numbers

        Parameters
        ----------
        ind : str or int
            dataset index

        src : str
            the name of the placeholder with data

        dst : str
            the name of the placeholder in witch the result will be recorded"""
        _ = args, kwargs
        i = self.get_pos(None, src, ind)
        label = getattr(self, src)[i]
        more_label = np.array([int(i) for i in label.replace(',', '')] + [None])[:-1]
        getattr(self, dst)[i] = more_label

    def _reraise_exceptions(self, results):
        """Reraise all exceptions in the ``results`` list.
        Parameters
        ----------
        results : list
            Post function computation results.

        Raises
        ------
        RuntimeError
            If any paralleled action raised an ``Exception``.
        """
        if any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    def _assemble_load(self, results, *args, **kwargs):
        """Assemble batch use ``results``

        Parameters
        ----------
        results : array
            loaded image

        Returns
        -------
        self"""
        _ = args, kwargs
        self._reraise_exceptions(results)
        components = kwargs.get('components', None)
        if components is None:
            components = self.components
        for comp, data in zip(components, zip(*results)):
            data = np.array(data)
            setattr(self, comp, data)
        return self
