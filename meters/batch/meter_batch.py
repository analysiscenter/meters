# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
import numpy as np
import tensorflow as tf

from scipy.misc import imresize
from ..dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed, DatasetIndex

class MeterBatch(ImagesBatch):
    """Batch class for meters"""
    components = 'images', 'labels', 'coordinates', 'indices', 'background', 'predicted_bb', 'confidence', 'cropped_images'


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
        else:
            if getattr(self, dst) == None:
                setattr(self, dst, np.array([None] * len(self.index)))
        return self.indices

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
        # print('crop_from_bbox begin')

        image = self.get(ix, src)
        coord_str = self.get(ix, 'coordinates')
        x, y, width, height = list(map(int, coord_str.split()))
        i = self.get_pos(None, src, ix)
        dst_data = image[y:y+height, x:x+width].copy()
        image[y:y+height, x:x+width]
        getattr(self, dst)[i] = dst_data


    @action
    @inbatch_parallel(init='_init_component', src='images', dst='background', target='threads')
    def crop_background(self, ix, src='images', dst='background', new_size=(32, 192, 3)):
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
        x, y, width, height = list(map(int, coord_str.split()))
        image[y:y+height, x:x+width] = image[y - height:y, x:x+width]

        new_height, new_width = new_size[0], new_size[1]
        y_crop = np.random.randint(0, image.shape[0] - new_height)
        x_crop = np.random.randint(0, image.shape[1] - new_width)
        dst_data = image[y_crop: y_crop + new_height, x_crop: x_crop + new_width, :]
        i = self.get_pos(None, src, ix)
        try:
            getattr(self, dst)[i] = dst_data
        except Exception as e:
            print(e, 'crop_background failed')
            raise ValueError

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
        batch = MeterBatch(DatasetIndex(np.arange(len(self.labels.reshape(-1)))))
        batch.labels = self.labels.reshape(-1)
        batch.background = np.tile(self.background, len(self.labels))
        batch.predicted_bb = []
        batch.cropped_images = []
        numbers = np.array([None] * len(self.index))
        for i, image in enumerate(self.display):
            # [None] is added because numpy can not automaticlly create an array with `object` type.
            numbers[i] = np.array([*np.array_split(image, n_digits, axis=1) + [None]])[:-1]

        batch.images = np.concatenate(numbers)
        return batch


    @action
    @inbatch_parallel(init='_init_component', dst='background', post='assemble', components=('images', 'labels', 'coordinates', 'confidence'))
    def generate_data(self, ix, dst='background', n_digits=8, normalize=True):
        ''' Generate image with n_digits random MNIST digits om it
        Parameters
        ----------
        image : np.array
        '''
        image = self.get(ix, 'images')
        height, width = image.shape[:2]
        new_size = (height, np.round(width * n_digits * 1.5), 3)
        try:
            canvas = self.get(ix, 'background')
        except Exception as e:
            print('failed on ', ix, e)
            canvas = np.zeros(new_size) + np.mean(image)

        random_indices = np.random.choice(self.images.shape[0], n_digits)
        random_images = [np.squeeze(self.images[i]) for i in random_indices]
        labels = np.array([self.labels[i] for i in random_indices]).reshape(-1)
        coordinates = []
        right_y = 0
        confidence = [0] * n_digits
        left_x = 0.0
        for index, random_image in enumerate(random_images):
            height = random_image.shape[0]
            width = random_image.shape[1]
            left_y = np.random.randint(right_y, right_y + np.round(0.5 * width))
            right_y = left_y + width
            if np.random.binomial(1, 0.7) == 1:
                canvas[:, left_y:right_y, :] = random_image
                confidence[index] = 1
            if normalize:
                left_y /= new_size[1]
                width /= new_size[1]
                coordinates.append([left_x, left_y,  1.0, width])
            else:
                width = float(width)
                coordinates.append([left_x, left_y, height, width])
        return canvas, labels, coordinates, confidence

    @action
    @inbatch_parallel(init='indices', post='assemble', components='cropped_images')
    def crop_predictions(self, idx, n_digits=8, confidence_treshold=0.5, new_size=(32, 16)):
        """Split image with ``n_digits`` numbers to ``n_digits`` images each with one number

        Parameters
        ----------
        n_digits : int
            number of digits on meter

        Returns
        ------
        self
        """
        # batch = MeterBatch(DatasetIndex(np.arange(len(self.labels.reshape(-1)))))
        # batch.labels = self.labels.reshape(-1)
        # batch.coordinates = self.coordinates
        # binary_confidence = (self.confidence.reshape(-1) > confidence_treshold).astype(int)
        coordinates = self.get(idx, 'predicted_bb')
        try:
            print('predicted_bb ', coordinates)
        except Exception as e:
            print("HERE", e)
        # print(coordinates.shape, 'coordinates')
        denormalized_coordinates = self.denormalize_bb(self.images.shape[1:3], coordinates)
        # print(denormalized_coordinates.shape, 'denormalized_coordinates')
        images = self.get(idx, 'images')
        cropped_images = []
        for i in range(n_digits):
            current_coords = denormalized_coordinates[i]
            print(current_coords.shape, i, 'coords shape')
            print(images.shape, 'images')
            cropped_images.append(imresize(images[current_coords[0]:current_coords[2], \
                                         current_coords[1]:current_coords[3]], new_size))

            
        # self.cropped_images = np.stack(cropped_images, axis=1)
        cropped_images = np.stack(cropped_images, axis=1)
        print(cropped_images.shape, 'CROPPED SHAPE')

        return cropped_images

    def denormalize_bb(self, img_size, coordinates, n_digits=8):
        height, width = img_size
        coordinates = coordinates.reshape(-1, 4)
        boarders = np.ones((coordinates.shape[0]))
        scales = [height, height, width, width]
        for i in range(4):
            coordinates[:, i] = np.minimum(coordinates[:, i], boarders)
            coordinates[:, i] *= scales[i]
        coordinates[:, 2] += coordinates[:, 0]
        coordinates[:, 3] += coordinates[:, 1]
        return coordinates.astype(np.int64)

    @action
    @inbatch_parallel(init='indices', post='assemble', components='labels')
    def one_hot(self, ind):
        """ One hot encoding for labels
        Parameters
        ----------
        ind : numpy.uint8
            index
        Returns
        -------
            One hot labels"""
        label =  self.get(ind, 'labels')
        one_hot = np.zeros(10)
        one_hot[label] = 1
        return one_hot.reshape(-1)

    @action
    @inbatch_parallel(init='indices', post='assemble', src='labels', components='labels')
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
