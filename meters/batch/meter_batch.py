# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
from numbers import Number
from time import time

import numpy as np
import scipy as sp

from imageio import imwrite



from ..dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed, DatasetIndex
from ..dataset.dataset.batch_image import transform_actions

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, (xB - xA)) * max(0, (yB - yA))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = max(0,(boxA[2] - boxA[0])) * max(0,(boxA[3] - boxA[1]))
    boxBArea = max(0,(boxB[2] - boxB[0])) * max(0,(boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-10)

    # return the intersection over union value
    return iou

@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class MeterBatch(ImagesBatch):
    """Batch class for meters"""
    components = 'images', 'labels', 'coordinates', 'indices'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_stamps = {}


    def _check_bbox(self, bbox, image):
        lt = np.array(bbox[:2])
        rb = bbox[:2] + np.array(bbox[2:])
        shape = self._get_image_shape(image)[::-1]
        condition = (lt >= 0)* (lt <= shape) * (rb >= 0) * (rb <= shape) * (rb >= lt)
        return np.all(condition)


    def _normalize_bb_(self, image, bbox, **kwargs):
        bbox /= np.tile(self._get_image_shape(image)[::-1], 2)
        return image, bbox

    @action
    def record_time(self, stat_name='statistics', mode='record'):
        """
        Make a time stamp or record the difference between the previous
        one for a specified list in the pipeline

        Parameters
        ----------
        stat_name : str
                    name of the statistics in the pipeline for which the operation is conducted
        mode : {'record', 'diff'}
               if 'record' is given then the current time is recorded with the handler specified by `stat_name`
               if 'diff' is given then the difference between the current time and the last recorded time for
               the given handler (`stat_name`) is appended to `stat_name` in the pipeline

        Returns
        -------
        self : MNISTBatchTime

        """
        if mode == 'record':
            self._time_stamps[stat_name] = time()
#             if stat_name == 'resize_time':
#             print('record', stat_name, self._time_stamps)
        elif mode == 'diff':
#             if stat_name == 'resize_time':
            # print('diff', stat_name, time()-self._time_stamps[stat_name])
            self.pipeline.set_variable(
                stat_name,
                time()-self._time_stamps[stat_name],
                mode='append')
#             if stat_name == 'resize_time':
#                 print('array',self.pipeline.get_variable(stat_name))
        return self

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
    def debug(self, *args, **kwargs):
        images = self.get(None, 'images')
        print(images.shape)
        return self

    @action
    def iou(self, mode='set',**kwargs):
        iou_ = 0
        predictions = self.pipeline.get_variable('predictions')
        for i in range(len(self)):
            boxA, boxB = self.coordinates[i].copy(), predictions[i].copy()
            boxA[[2,3]] = np.maximum(0., boxA[[2,3]])
            boxB[[2,3]] = np.maximum(0., boxB[[2,3]])
            boxA[2] += boxA[0]
            boxA[3] += boxA[1]
            boxB[2] += boxB[0]
            boxB[3] += boxB[1]
            iou_ += iou(boxA, boxB)
        self.pipeline.set_variable('current_iou', iou_ / len(self), mode=mode)

        return self


    def _resize_bb_only_(self, bbox, original_shape, shape, **kwargs):
        factor = np.asarray(shape) / np.asarray(original_shape)
        return bbox * np.tile(factor[::-1], 2)

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

    @action
    @inbatch_parallel(init='indices')
    def imsave(self, ix, components, *args, **kwargs):
        image = self.get(ix, components)
        bbox = self.get(ix, 'coordinates')
        name = 'aug' + str(np.random.randint(10**16, 10**17))
        self.pipeline.set_variable('bboxes', {name: bbox}, mode='u')
        imwrite('../data/augmented_images/'+name+'.png',image)

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

        new_bbox=bbox.copy()

        left_top = np.linalg.solve(matrix, (new_bbox[1], new_bbox[0], 0, 1))[:2]
        left_bottom = np.linalg.solve(matrix, (new_bbox[1]+new_bbox[3], new_bbox[0], 0, 1))[:2]
        right_top = np.linalg.solve(matrix, (new_bbox[1], new_bbox[0]+new_bbox[2], 0, 1))[:2]
        right_bottom = np.linalg.solve(matrix, (new_bbox[1]+new_bbox[3], new_bbox[0]+new_bbox[2], 0, 1))[:2]

        right_bottom_bb = np.zeros(2)

        for i in (0, 1):
            new_bbox[1-i] = min(left_top[i], left_bottom[i], right_bottom[i], right_top[i])
            right_bottom_bb[1-i] = max(left_top[i], left_bottom[i], right_bottom[i], right_top[i])
        new_bbox[2], new_bbox[3] = right_bottom_bb - new_bbox[:2]

        if not self._check_bbox(new_bbox, image):
            return image, bbox
        return super()._affine_transform_(image, matrix=matrix, *args, **kwargs), new_bbox

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
        new_bbox = np.r_[np.array(bbox[:2]) + shift[1::-1], bbox[2:]]
        if not self._check_bbox(new_bbox, image):
            return image, bbox
        return super()._shift_(image, shift, **kwargs), new_bbox

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

        factor = (factor, factor) if isinstance(factor, Number) else factor
        shift = np.array(self._get_image_shape(image)) / 2
        scale_matrix = np.array([[factor[0],         0, 0, shift[0]-factor[0]*shift[0]],
                                 [        0, factor[1], 0, shift[1]-factor[1]*shift[1]],
                                 [        0,         0, 1,                           0],
                                 [        0,         0, 0,                           1]])
        return self._affine_transform_(image, bbox, scale_matrix, **kwargs)


    def _flip_(self, image, bbox, mode):
        """ Flips image.

        Parameters
        ----------
        mode : {'lr', 'ud'}
            - 'lr' - apply the left/right flip
            - 'ud' - apply the upside/down flip
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

        shape = self._get_image_shape(image)
        new_bbox = bbox.copy()
        if mode == 'ud':
            new_bbox[1] = shape[0] - bbox[1] - bbox[3]
        elif mode == 'lr':
            new_bbox[0] = shape[1] - bbox[0] - bbox[2]
        return super()._flip_(image, mode), new_bbox

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
