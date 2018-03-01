# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
import copy
import numpy as np
import tensorflow as tf
from scipy.special import expit
from scipy.misc import imresize
import matplotlib.pyplot as plt
# import sys
# sys.path.append('..//..//meters/')
from ..dataset.dataset import ImagesBatch, action, inbatch_parallel, any_action_failed, DatasetIndex

class MeterBatch(ImagesBatch):
    """Batch class for meters"""
    components = 'images', 'labels', 'coordinates', 'indices', 'background', 'predicted_bb', 'confidence', 'cropped_images', 'new_images'


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
        print(len(self.labels), 'batch labels')
        batch.labels = np.array(self.labels).reshape(-1)
        print(len(batch.labels), ' flatten batch labels')

        # print(self.background.shape, 'self.background shape')
        # batch.background = np.tile(self.background, n_digits)
        # print(batch.background.shape, 'batch.background shape')

        batch.predicted_bb = []
        batch.cropped_images = []
        numbers = np.array([None] * len(self.index))
        try:
            for i, image in enumerate(self.display):
                # [None] is added because numpy can not automaticlly create an array with `object` type.
                numbers[i] = np.array([*np.array_split(image, n_digits, axis=1) + [None]])[:-1]
        except Exception as e:
            print(e, 'for has failed')
        batch.images = np.concatenate(numbers)
        print('split digits done')
        return batch

    # @action
    # def dump_digits():


    @action
    def split_cropped(self, n_digits=8, new_size=(32, 16, 3)):
        batch = MeterBatch(DatasetIndex(np.arange(len(self.labels.reshape(-1)))))
        batch.labels = self.labels.reshape(-1)
        batch.images = self.cropped_images.reshape(-1, *new_size)
        # batch.images = np.tile(self.images, n_digits)
        print(batch.cropped_images.shape, 'batch cropped images shape')
        return batch

    @action
    @inbatch_parallel(init='indices', post='_assemble', components=('new_images', 'labels', 'coordinates', 'confidence'))
    def generate_data(self, ix, n_digits=8, normalize=True, scale_factor=1.2):
        ''' Generate image with n_digits random MNIST digits om it
        Parameters
        ----------
        image : np.array
        '''

        image = self.get(ix, 'images')
        canvas = copy.deepcopy(self.get(ix, 'background'))
        # canvas = canvas / 255. because imresize will return data in [0, 255]
        canvas_size = canvas.shape[:2]
        random_indices = np.random.choice(self.images.shape[0], n_digits)
        labels = np.array([self.labels[i] for i in random_indices]).reshape(-1)

        scale_factor = np.random.uniform(0.7, 1.5)
        height, width = (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor))
        random_images = [imresize(self.images[i], (height, width)) for i in random_indices]

        interval = np.random.randint(0, max(1e-5, (canvas_size[1] - width * 8) / 8))
        coordinates = []
        confidence = np.zeros((n_digits, 1))
        left_x = np.random.randint(0.0, canvas_size[0] - height)
        try:
            left_y = np.random.randint(low=0, high=max(1e-5, canvas_size[1] - (width + interval) * 8))
        except Exception as e:
            print(e, 'new_size ', canvas_size, ' (width + interval) * 8) ',  (width + interval) * 8)
        right_y = 0
        gap_number = int(np.random.normal(loc=0, scale=1.2))
        for index, random_image in enumerate(random_images):
            height = random_image.shape[0]
            width = random_image.shape[1]

            left_y = right_y + interval
            right_y = left_y + width

            if gap_number > 0 and index <= gap_number or gap_number < 0 and index >= n_digits + gap_number:
                try:
                    confidence[index, 0] = 0
                except Exception as e:
                    print('flag2', e)

            else:
                try:
                    canvas[left_x:left_x + height, left_y:right_y, :] = random_image
                except Exception as e:
                    print('canvas crop in generate data', e, random_image.shape, canvas.shape, left_x, left_y, right_y)

                try:
                    confidence[index, 0] = 1
                except Exception as e:
                    print('flag1', e)
            
            if normalize:
                norm_left_y = left_y / canvas_size[1]
                norm_left_x = left_x / canvas_size[0]
                norm_width = width / canvas_size[1]
                norm_height = height / canvas_size[0]
                coordinates.append([norm_left_x, norm_left_y,  norm_height, norm_width])
            else:
                new_width = float(width)
                coordinates.append([left_x, left_y, height, new_width])
        print('inside generate_data: canvas.shape is', canvas.shape)
        return canvas, labels, coordinates, confidence




    @action
    @inbatch_parallel(init='indices', post='_assemble', components='cropped_images')
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
        # print('enter crop detection')
        print('enter crop_predictions')
        try:
            predictions = self.get(idx, 'predicted_bb')
            print('predictions shape ', predictions.shape)
        except Exception as e:
            print('212', e)
        try:
            real_coordinates = self.get(idx, 'coordinates')
        except Exception as e:
            print('217', e)
            raise ValueError

        predictions = predictions.reshape((-1, 5))
        print('reshaped predictions SHAPE ', predictions.shape)


        coordinates = predictions[:, :4]
        predicted_confidence = predictions[:, 4:5]

        left_corners = [coordinates[i][0] for i in range(n_digits)]
        sorted_indices = np.argsort(left_corners)
        coordinates = coordinates[sorted_indices]
        predicted_confidence = expit(predicted_confidence[sorted_indices])

        denormalized_coordinates = self.denormalize_bb(self.images.shape[1:3], coordinates)
        # print(denormalized_coordinates.shape, 'denormalized_coordinates')
        try:
            images = self.get(idx, 'images')
        except Exception as e:
            print('get images', e)
        cropped_images = []
        for i in range(n_digits):
            print('pred conf i', i , predicted_confidence[i])
            try:
                if predicted_confidence[i, 0] < confidence_treshold:
                    continue
            except Exception as e:
                print('vsee plohaa', e)
                raise ValueError    
            current_coords = denormalized_coordinates[i]
            # print('i SDSD', i, '  denormalized_coordinates[i] ', coordinates[i])
            # print('i SDSD', i, '  real_coordinates[i] ', real_coordinates[i])

            # print(current_coords.shape, i, 'coords shape')
            # print(images.shape, 'images')
            try:
                current = images[current_coords[0]:current_coords[2], \
                                         current_coords[1]:current_coords[3]]
            except Exception as e:
                print('SLICING ERROR', current_coords, e)

            try:
                cropped_images.append(imresize(current, new_size))
            except Exception as e:
                cropped_images.append(np.zeros((new_size[0], new_size[1], 3)))
                print('RESIZE EROOR', 'CURRENT ', current.shape)
                print('start ', current, e, 'end')
                pass
        # try:
        #     print('predicted_bb ', coordinates)
        # except Exception as e:
        #     print("HERE", e)

            
        # self.cropped_images = np.stack(cropped_images, axis=1)
        try:
            cropped_images = np.stack(cropped_images, axis=0)
        except Exception as e:
            print(cropped_images, 'stack cropped_images ', e)
        # print(cropped_images.shape, 'CROPPED SHAPE')
        return cropped_images

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
    @inbatch_parallel(init='indices', post='_assemble', components='labels')
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
