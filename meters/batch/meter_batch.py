# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
import numpy as np

from ..dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed, DatasetIndex

class MeterBatch(ImagesBatch):
    """Batch class for meters"""
    components = 'resized_images', 'images', 'labels', 'coordinates', 'indices', 'pred_coordinates'

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
    @inbatch_parallel(init='indices', src='images', post='assemble')
    def normalize_images(self, ix, src='images'):
        """ Normalize pixel values to (0, 1). """
        image = self.get(ix, src)
        image /= 255.
        return (image,)

    @action
    @inbatch_parallel(init='indices', post='_assemble', components='pred_coordinates')
    def get_global_coordinates(self, ix, src='pred_coordinates', img='images'):
        """Recalculates from relative coordinates to global.

        Parameters
        ----------
        src : str
            the name of the component with relarive coordinates
        img : str
            the name of the component with images

        Returns
        -------
        self
        """
        coordinates = self.get(ix, src)
        global_coord = np.maximum(0, coordinates * np.tile(self.get(ix, img).shape[1::-1], 2))
        return (list(map(np.int32, global_coord)),)

    @action
    @inbatch_parallel(init='_init_component', src='images', dst='display', target='threads')
    def crop_from_bbox(self, ix, src='images', dst='display', component_coord='coordinates'):
        """Crop area from an image using ``coordinates`` attribute

        Parameters
        ----------
        src : str
            data component's name
        dst : str
            the name of the component where the result will be recorded
        component_coord : str
            the name of the component with coordinates of the display with digits

        Returns
        -------
        self
        """
        image = self.get(ix, src)
        x, y, width, height = self.get(ix, component_coord)
        i = self.get_pos(None, src, ix)
        dst_data = image[y:y+height, x:x+width].copy()
        getattr(self, dst)[i] = dst_data

    @action
    def split_to_digits(self, n_digits=8, is_training=False):
        """Split image with ``n_digits`` numbers to ``n_digits`` images each with one number

        Parameters
        ----------
        n_digits : int
            number of digits on meter

        Returns
        ------
        self
        """
        batch = ImagesBatch(DatasetIndex(np.arange(n_digits * self.images.shape[0])))
        if is_training:
            batch.labels = self.labels.reshape(-1)
            if isinstance(batch.labels[0], list):
                batch.labels = np.concatenate(batch.labels)
        numbers = np.array([None] * len(self.index))
        for i, image in enumerate(self.display):
            # [None] is added because numpy can not automaticlly create an array with `object` type.
            numbers[i] = np.array([*np.array_split(image, n_digits, axis=1) + [None]])[:-1]

        batch.images = np.concatenate(numbers)
        return batch

    @action
    @inbatch_parallel(init='indices', post='_assemble', src='labels', components='labels')
    def split_labels(self, ix, src='labels'):
        """Splits labels from strig to list with separate numbers

        Parameters
        ----------
        src : str
            the name of the component with data

        Returns
        -------
        self
        """
        i = self.get_pos(None, src, ix)
        label = getattr(self, src)[i]
        more_label = list(map(int, label.replace(',', '')))
        return (more_label,)

    def _reraise_exceptions(self, results):
        """Reraise all exceptions in the ``results`` list

        Parameters
        ----------
        results : list
            Post function computation results

        Raises
        ------
        RuntimeError
            If any paralleled action raised an exception
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
