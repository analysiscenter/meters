# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
import numpy as np
import scipy as sp

from ..dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed, DatasetIndex
from ..dataset.dataset.batch_image import transform_actions

@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class MeterBatch(ImagesBatch):
    """Batch class for meters"""
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

    def _convert_bbox_(self, bbox):
        """ Convert bounding box's coordinates to ``int``.

        Parameters
        ----------
        src : str
            Component to get bounding boxes from. Default is 'images'.
        dst : str
            Component to write bounding boxes to. Default is 'images'.

        Returns
        -------
        self
        """

        return [int(x) for x in bbox.split(', ')]

    def _resize_bb_(self, image, bbox, shape, **kwargs):
        """ Resizes an image.

        Does the same thing as scipy.misc.imresize

        Parameters
        ----------
        shape : sequence
            Shape of the resulting image in the form (rows, columns).
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.

        Returns
        -------
        self
        """

        factor = np.asarray(shape) / np.asarray(self._get_image_shape(image))
        return super()._resize_(image, shape, **kwargs), bbox * np.tile(factor[::-1], 2)

    def _remove_background_(self, image):
        """ Removes white stripes from the edges of an image.

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.

        Returns
        -------
        self
        """

        rows_mean, columns_mean = image.mean((0, 2)), image.mean((1, 2))
        rows_nonwhite = np.argwhere(columns_mean < 255)
        columns_nonwhite = np.argwhere(rows_mean < 255)

        left_top = (rows_nonwhite[0][0], columns_nonwhite[0][0])
        right_bottom = (rows_nonwhite[-1][0]+1, columns_nonwhite[-1][0]+1)

        return image[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]].copy()

    def _affine_transform_(self, image, bbox, matrix, *args, **kwargs):
        """ Perfoms affine transformation.

        Does the same thing as scipy.ndimage.affine_transform

        Parameters
        ----------
        matrix : array_like
            matrix of the transformation. Its shape must be (4, 4)
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        Returns
        -------
        self
        """

        bbox=bbox.copy()

        left_top = np.linalg.solve(matrix, (bbox[1], bbox[0], 0, 1))[:2]
        left_bottom = np.linalg.solve(matrix, (bbox[1]+bbox[3], bbox[0], 0, 1))[:2]
        right_top = np.linalg.solve(matrix, (bbox[1], bbox[0]+bbox[2], 0, 1))[:2]
        right_bottom = np.linalg.solve(matrix, (bbox[1]+bbox[3], bbox[0]+bbox[2], 0, 1))[:2]

        right_bottom_bb = np.zeros(2)

        for i in (0, 1):
            bbox[1-i] = min(left_top[i], left_bottom[i], right_bottom[i], right_top[i])
            right_bottom_bb[1-i] = max(left_top[i], left_bottom[i], right_bottom[i], right_top[i])
        bbox[2], bbox[3] = right_bottom_bb - bbox[:2]

        return super()._affine_transform_(image, matrix=matrix, *args, **kwargs), bbox

    def _rotate_(self, image, bbox, angle, **kwargs):
        """ Rotates an image

        Parameters
        ----------
        angle : float
            angle of rotation in degrees
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        Returns
        -------
        self
        """

        angle *= np.pi / 180
        cos, sin = np.cos(angle), np.sin(angle)
        shift = np.array(self._get_image_shape(image)) / 2
        rotation_matrix = np.array([[cos, -sin, 0, -cos*shift[0]+shift[0]+shift[1]*sin],
                                    [sin,  cos, 0, -cos*shift[1]+shift[1]-shift[0]*sin],
                                    [  0,    0, 1,                                   0],
                                    [  0,    0, 0,                                   1]])
        return self._affine_transform_(image, bbox, rotation_matrix, **kwargs)

    def _shift_(self, image, bbox, shift, **kwargs):
        """ Shifts an image

        Parameters
        ----------
        shift : sequence
            shift's value in the form (row, columns)
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        Returns
        -------
        self
        """

        return super()._shift_(image, shift), np.r_[np.array(bbox[:2]) + shift[1::-1], bbox[2:]]

    def _scale_(self, image, bbox, factor, preserve_shape, **kwargs):
        """ Scale the content of each image in the batch.

        Resulting shape is obtained as original_shape * factor.

        Parameters
        -----------
        factor : float, sequence
            resulting shape is obtained as original_shape * factor
            - float - scale all axes with the given factor
            - sequence (factor_1, factort_2, ...) - scale each axis with the given factor separately

        preserve_shape : bool
            whether to preserve the shape of the image after scaling

        origin : {'center', 'top_left', 'random'}, sequence
            Relevant only if `preserve_shape` is True.
            Position of the scaled image with respect to the original one's shape.
            - 'center' - place the center of the rescaled image on the center of the original one and crop
                         the rescaled image accordingly
            - 'top_left' - place the upper-left corner of the rescaled image on the upper-left of the original one
                           and crop the rescaled image accordingly
            - 'random' - place the upper-left corner of the rescaled image on the randomly sampled position
                         in the original one. Position is sampled uniformly such that there is no need for cropping.
            - sequence - place the upper-left corner of the rescaled image on the given position in the original one.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        Returns
        -------
        self
        """

        shift = np.array(self._get_image_shape(image)) / 2
        scale_matrix = np.array([[factor[0],         0, 0, shift[0]-factor[0]*shift[0]],
                                 [        0, factor[1], 0, shift[1]-factor[1]*shift[1]],
                                 [        0,         0, 1,                           0],
                                 [        0,         0, 0,                           1]])
        return self._affine_transform_(image, bbox, scale_matrix, **kwargs)

    @action
    @inbatch_parallel(init='_init_component', src='images', dst='display', target='threads')
    def crop_from_bbox(self, ix, src='images', dst='display'):
        """Crop area from an image using ``coordinates`` attribute

        Parameters
        ----------
        src : str
            data component's name
        dst : str
            the name of the component where the result will be recorded

        Returns
        -------
        self
        """

        image = self.get(ix, src)
        coord_str = self.get(ix, 'coordinates')
        x, y, width, height = list(map(int, coord_str.split(', ')))
        i = self.get_pos(None, src, ix)
        dst_data = image[y:y+height, x:x+width].copy()
        getattr(self, dst)[i] = dst_data

    @action
    def split_to_digits(self, n_digits=8):
        """Split image with ``n_digits`` numbers to ``n_digits`` images each with one number

        Parameters
        ----------
        n_digits : int
            number of digits on meter

        Returns
        ------
        self
        """

        batch = ImagesBatch(DatasetIndex(np.arange(len(self.labels.reshape(-1)))))
        batch.labels = self.labels.reshape(-1)
        numbers = np.array([None] * len(self.index))
        for i, image in enumerate(self.display):
            # [None] is added because numpy can not automaticlly create an array with `object` type.
            numbers[i] = np.array([*np.array_split(image, n_digits, axis=1) + [None]])[:-1]

        batch.images = np.concatenate(numbers)
        return batch

    @action
    @inbatch_parallel(init='indices', post='_assemble', src='labels', dst='labels')
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
        return more_label

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
