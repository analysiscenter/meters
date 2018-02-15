# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
import copy
import numpy as np
import tensorflow as tf
from scipy.special import expit
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
        # print('crop_background generate_data')

        image = self.get(ix, src)

        coord_str = self.get(ix, 'coordinates')
        x, y, width, height = list(map(int, coord_str.split()))
        background = copy.deepcopy(image)
        background[y:y+height, x:x+width] = image[y - height:y, x:x+width]

        new_height, new_width = new_size[0], new_size[1]
        y_crop = np.random.randint(0, image.shape[0] - new_height)
        x_crop = np.random.randint(0, image.shape[1] - new_width)
        dst_data = background[y_crop: y_crop + new_height, x_crop: x_crop + new_width, :]
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
        # print(self.background.shape, 'self.background shape')
        batch.background = np.tile(self.background, n_digits)
        # print(batch.background.shape, 'batch.background shape')

        batch.predicted_bb = []
        batch.cropped_images = []
        numbers = np.array([None] * len(self.index))
        for i, image in enumerate(self.display):
            # [None] is added because numpy can not automaticlly create an array with `object` type.
            numbers[i] = np.array([*np.array_split(image, n_digits, axis=1) + [None]])[:-1]

        batch.images = np.concatenate(numbers)
        return batch


    @action
    def split_cropped(self, n_digits=8, new_size=(32, 16, 3)):
        batch = MeterBatch(DatasetIndex(np.arange(len(self.labels.reshape(-1)))))
        batch.labels = self.labels.reshape(-1)
        batch.images = self.cropped_images.reshape(-1, *new_size)
        # batch.images = np.tile(self.images, n_digits)
        print(batch.cropped_images.shape, 'batch cropped images shape')
        return batch

    @action
    @inbatch_parallel(init='indices', post='assemble', components=('images', 'labels'))
    def gen_classification_data(self, ix, background_proportion=0.1):
        ''' Generate data for training classification task to classify 10 digits and
        background class. Method takes current batch with digits after split_cropped and randomly
        adds digit image for image with random meters' background.
        Parameters
        ---------
        ix : index of current element in batch
        Returns
        -------
        image, label
        '''
        label = self.get(ix, 'labels')
        image = self.get(ix, 'images')
        if np.random.binomial(1, background_proportion) == 1:
            crop_height = image.shape[0]
            crop_width = image.shape[1]
            background = self.get(ix, 'background')
            x = np.random.randint(0, background.shape[0] - crop_height)
            y = np.random.randint(0, background.shape[1] - crop_width)
            image = background[x: x + crop_height, y:y + crop_width, :]
            label = np.hstack((np.zeros((10)), [1]))
        else:
            label = np.hstack((label, [0]))
        return image, label

    @action
    @inbatch_parallel(init='indices', post='assemble', components=('images', 'labels', 'coordinates', 'confidence'))
    def generate_data(self, ix, n_digits=8, normalize=True, prob=0.7):
        ''' Generate image with n_digits random MNIST digits om it
        Parameters
        ----------
        image : np.array
        '''
        # print('enter generate_data')

        image = self.get(ix, 'images')
        height, width = image.shape[:2]
        try:
            canvas = copy.deepcopy(self.get(ix, 'background'))
        except Exception as e:
            print('failed on ', ix, e)
        new_size = canvas.shape[:2]

        random_indices = np.random.choice(self.images.shape[0], n_digits)
        random_images = [np.squeeze(self.images[i]) for i in random_indices]
        labels = np.array([self.labels[i] for i in random_indices]).reshape(-1)
        coordinates = []
        confidence = np.zeros((n_digits, 1))

        left_x = np.random.randint(0.0, new_size[0] - height)
        right_y = 0
        for index, random_image in enumerate(random_images):
            height = random_image.shape[0]
            width = random_image.shape[1]

            left_y = np.random.randint(right_y, right_y + np.round(0.5 * width))
            right_y = left_y + width
            if np.random.binomial(1, prob) == 1:
                canvas[left_x:left_x + height, left_y:right_y, :] = random_image
                try:
                    confidence[index, 0] = 1
                except Exception as e:
                    print('flag1', e)
            else:
                try:
                    confidence[index, 0] = 0
                except Exception as e:
                    print('flag2', e)
            if normalize:
                left_y /= new_size[1]
                width /= new_size[1]
                new_left_x = left_x / new_size[0]
                height /= new_size[0]
                coordinates.append([new_left_x, left_y,  height, width])
            else:
                width = float(width)
                coordinates.append([left_x, left_y, height, width])
        return canvas, labels, coordinates, confidence

    @action
    @inbatch_parallel(init='indices', post='assemble', components=('cropped_images', 'labels'))
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
        try:
            predictions = self.get(idx, 'predicted_bb')
            print
            labels = self.get(idx, 'labels')
            confidence = self.get(idx, 'confidence')
            # print('predictions shape ', predictions.shape)
        except Exception as e:
            print('212', e)
        try:
            real_coordinates = self.get(idx, 'coordinates')
        except Exception as e:
            print('217', e)
            raise ValueError

        predictions = predictions.reshape((-1, 4))
        coordinates = predictions[:, :4]

        left_corners = [coordinates[i][0] for i in range(n_digits)]
        sorted_indices = np.argsort(left_corners)
        coordinates = coordinates[sorted_indices]
        denormalized_coordinates = self.denormalize_bb(self.images.shape[1:3], coordinates)
        try:
            images = self.get(idx, 'images')
        except Exception as e:
            print('get images', e)
        cropped_images = []

        for i in range(n_digits):
            current_coords = denormalized_coordinates[i]
            print(current_coords)
            try:
                current = images[current_coords[0]:current_coords[2], \
                                         current_coords[1]:current_coords[3]]
            except Exception as e:
                print('SLICING ERROR', current_coords, e)

            try:
                cropped_images.append(imresize(current, new_size))
                if confidence[i] == 1:
                    labels[i] = np.hstack((labels[i], [0]))
                else:
                    labels[i] = np.hstack((np.zeros((10)), [1]))
            except Exception as e:
                cropped_images.append(np.zeros((new_size[0], new_size[1], 3)))
                print('RESIZE EROOR', 'CURRENT ', current, e)
                pass
            
        # self.cropped_images = np.stack(cropped_images, axis=1)
        try:
            cropped_images = np.stack(cropped_images, axis=0)
        except Exception as e:
            print(cropped_images, 'stack cropped_images ', e)
        try:
            labels = labels.reshape((-1, 11))
        except Exception as e:
            print('labels reshape fail', e)
        # print(cropped_images.shape, 'CROPPED SHAPE')
        return cropped_images, labels

    def denormalize_bb(self, img_size, coordinates, n_digits=8):
        height, width = img_size
        coordinates = copy.deepcopy(coordinates)
        coordinates = coordinates.reshape(-1, 4)
        max_boarders = np.ones((coordinates.shape[0]))
        min_boarders = np.zeros((coordinates.shape[0]))
        scales = [height, height, width, width]
        for i in range(4):  
            coordinates[:, i] = np.minimum(coordinates[:, i], max_boarders)
            coordinates[:, i] = np.maximum(coordinates[:, i], min_boarders)
            coordinates[:, i] *= scales[i]
        coordinates[:, 2] += coordinates[:, 0]
        coordinates[:, 3] += coordinates[:, 1]
        # print(coordinates)
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
