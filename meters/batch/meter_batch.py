# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
import numpy as np

from ..dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed, DatasetIndex

class MeterBatch(ImagesBatch):
    """Class to create meters' batch"""
    components = 'images', 'labels', 'coordinates', 'indices'

    def _init_component(self, **kwargs):
        """Create a new attribute with the name specified by ``kwargs['dst']``,
        preallocate memory for it and return batch's indices

        Returns
        -------
        array with indices from batch
        """
        dst = kwargs.get('dst')
        if dst is None:
            raise KeyError('dst argument must be specified')
        if not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self.index)))
        return self.indices

    @action
    @inbatch_parallel(init='_init_component', src='images', dst='display', target='threads')
    def crop_to_display(self, ix, src='images', dst='display'):
        """Crop area from image using ``coordinates`` attribute

        Parameters
        ----------
        ix : str or int
            dataset's index
        src : str
            data placeholder's name
        dst : str
            the name of the placeholder's in witch the result will be recorded
        """
        image = self.get(ix, src)
        coord_str = self.get(ix, 'coordinates')
        x, y, width, height = list(map(int, coord_str.split()))
        i = self.get_pos(None, src, ix)
        dst_data = image[y:y+height, x:x+width].copy()
        getattr(self, dst)[i] = dst_data

    @action
    def split_to_digits(self, n_digits=8):
        """Crop image with ``n_digits`` number s to ``n_digits`` images with one number

        Parameters
        ----------
        n_digits : int
            number of digits on meter
        Return
        ------
        a new instance of ImagesBatch class
        """
        batch = ImagesBatch(DatasetIndex(np.arange(len(self.labels.reshape(-1)))))
        batch.labels = self.labels.reshape(-1)
        numbers = np.array([None] * len(self.index))
        for i, image in enumerate(self.display):
            # We add [None] because numpy can not automaticlly create an array with the object type.
            numbers[i] = np.array([*np.array_split(image, n_digits, axis=1) + [None]])[:-1]

        batch.images = np.concatenate(numbers)
        return batch

    @action
    @inbatch_parallel(init='indices', post='assemble', src='labels', components='labels')
    def split_labels(self, ix, src='labels'):
        """Splited labels from strig to list with separate numbers

        Parameters
        ----------
        ix : str or int
            dataset's index
        src : str
            the name of the placeholder's with data
        Returns
        -------
        array with int digits
        """
        i = self.get_pos(None, src, ix)
        label = getattr(self, src)[i]
        more_label = list(map(int, label.replace(',', '')))
        return more_label

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
        """Assemble batch using ``results``

        Parameters
        ----------
        results : array
            loaded data
        Returns
        -------
        self
        """
        _ = args
        self._reraise_exceptions(results)
        components = kwargs.get('components', None)
        if components is None:
            components = self.components
        for comp, data in zip(components, zip(*results)):
            data = np.array(data)
            setattr(self, comp, data)
        return self
