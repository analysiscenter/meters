# pylint: disable=attribute-defined-outside-init
"""Batch class for water meter task"""
import copy
import cv2
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
    components = 'images', 'labels', 'coordinates', 'indices', 'background', 'predicted_bb', 'confidence', 'cropped_images', \
                 'new_images', 'cropped_labels', 'resized_images', 'pred_coordinates'

    @property
    def target(self):
        result = np.concatenate((self.confidence, self.coordinates), axis=-1)
        return result

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


    def denormalize_bboxes(self, data, height, width):
        """ Converts values of bbox to global coordinates and changes layout to 
            [upper_left_y, upper_left_x, lower_right_y, lower_right_x].
        Parameters
        ----------
        data : np.array
            normalized bboxes' coordinates in format [left_x, left_y, height, width]
        height : int
            height of the image
        width : int
            width of the image
        Returns
        -------
        denormalized bbox
        """
        # print('enter')
        data = copy.deepcopy(data.reshape((-1, 4)))
        data[:, 0] *= height
        data[:, 2] *= height
        data[:, 1] *= width
        data[:, 3] *= width
        
        data[:, 2] += data[:, 0]
        data[:, 3] += data[:, 1]
        # print('exit')
        return np.vstack([data[:, 1], data[:, 0], data[:, 3], data[:, 2]]).T

    def normalize_bboxes(self, data, height, width):
        """ Converts values of bbox to relative coordinates and changes an order of axes.
        Parameters
        ----------
        data : np.array
            global bboxes' coordinates in format 
            [upper_left_y, upper_left_x, lower_right_y, lower_right_x]
        height : int
            height of the image
        width : int
            width of the image
        Returns
        -------
        normalized bbox in format [left_x, left_y, height, width]
        """
        data = copy.deepcopy(data.reshape((-1, 4)))
        data[:, 0] /= float(width)
        data[:, 2] /= float(width)
        data[:, 1] /= float(height)
        data[:, 3] /= float(height)
        
        data[:, 2] = data[:, 2] - data[:, 0]
        data[:, 3] = data[:, 3] - data[:, 1]
        return np.vstack([data[:, 1], data[:, 0], data[:, 3], data[:, 2]]).T

    @action
    @inbatch_parallel(init='indices', post='_assemble', components=('new_images', 'coordinates'))
    def hommography_transform(self, ix, eps=10, resize_prob=0.9):
        image = self.get(ix, 'new_images')
        bboxes = self.get(ix, 'coordinates')
        digit_size = (bboxes.reshape((8, 4))[0, 3] * image.shape[1], bboxes.reshape((8, 4))[0, 2] * image.shape[0])
        denorm_boxes = self.denormalize_bboxes(bboxes, image.shape[0], image.shape[1])
        pts_src = np.array([denorm_boxes[0, 0:2], [denorm_boxes[0, 0], denorm_boxes[0, 3]],
                            denorm_boxes[-1, :2], [denorm_boxes[-1, 0], denorm_boxes[-1, 3]]])
        pts_dst = copy.deepcopy(pts_src)
        case = np.random.randint(0, 3)
        direcrtion = np.random.randint(0, 1)

        direcrtion = 1
        if direcrtion == 0:
            pts_dst[case, direcrtion] += digit_size[direcrtion] * np.minimum(np.abs(np.random.normal(0.2, 0.1)), np.array([0.5]))
        else:
            pts_dst[case, direcrtion] += digit_size[direcrtion] * np.minimum(np.abs(np.random.normal(0.2, 0.1)), np.array([0.5]))

        h, status = cv2.findHomography(pts_src, pts_dst)
        im_out = cv2.warpPerspective(image, h, (image.shape[1],image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        new_bboxes = np.dot(h, np.concatenate((denorm_boxes.reshape((8, 2, 2)) , np.ones((8, 2, 1))), axis=2).transpose(0, 2, 1))
        result = (new_bboxes / new_bboxes[2])[:2, :, :].transpose(1, 2, 0)
        
        if np.random.uniform(0, 1) < resize_prob:
            low_y, low_x = np.minimum(result[0, 0, :].astype(np.int16), result[-1, 0, :].astype(np.int16))
            low_y, low_x = np.maximum(np.array([low_y, low_x]) - np.random.randint(0, eps, 2), np.array([0, 0]))
            high_y, high_x = np.maximum(result[-1, 1, :].astype(np.int16), result[0, 1, :].astype(np.int16))
            high_y, high_x = np.minimum(np.array([high_y, high_x]) + np.random.randint(0, eps, 2), np.asarray(im_out.shape[:2])[::-1])

            new_im_out = im_out[low_x: high_x, low_y: high_y]
            result[:, :, 0] -= low_y
            result[:, :, 1] -= low_x
            factor = np.asarray(im_out.shape[:2]) / np.asarray(new_im_out.shape[:2])
            result = (np.reshape(result, (-1, 4)) * np.tile(factor[::-1], 2)).reshape((-1, 2, 2))
            im_out = imresize(new_im_out, image.shape) / 255.
        normalized_result = self.normalize_bboxes(result, image.shape[0], image.shape[1])
        return im_out, normalized_result
    
    # @action
    # @inbatch_parallel(init='_init_component', src='images', dst='display', target='threads')
    # def crop_from_bbox(self, ix, src='images', dst='display'):
    #     """Crop area from an image using ``coordinates`` attribute

    #     Parameters
    #     ----------
    #     src : str
    #         data component's name
    #     dst : str
    #         the name of the component where the result will be recorded

    #     Returns
    #     -------
    #     self
    #     """
    #     # print('crop_from_bbox begin')

    #     image = self.get(ix, src)
    #     coord_str = self.get(ix, 'coordinates')
    #     x, y, width, height = list(map(int, coord_str.split()))
    #     i = self.get_pos(None, src, ix)
    #     dst_data = image[y:y+height, x:x+width].copy()
    #     image[y:y+height, x:x+width]
    #     getattr(self, dst)[i] = dst_data

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
    def put_dummies(self, num_digits=8):
        batch_size = self.new_images.shape[0]
        self.confidence = np.zeros((batch_size, num_digits, 1))
        self.coordinates = np.zeros((batch_size, num_digits, 4))
        return self

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
        batch = MeterBatch(DatasetIndex(np.arange(len(self.cropped_labels.reshape(-1)))))
        batch.cropped_labels = self.cropped_labels.reshape(-1)
        try:
            batch.cropped_images = self.cropped_images.reshape(-1, *new_size)
        except Exception as e:
            print('exception: ', e)
            print('batch.cropped_images ', self.cropped_images)
        # batch.labels = np.tile(self.labels, n_digits)
        # batch.images = np.tile(self.images, n_digits)
        # print(batch.cropped_images.shape, 'batch cropped images shape')
        return batch

    @action
    @inbatch_parallel(init='indices', post='_assemble', components=('new_images', 'labels', 'coordinates', 'confidence'))
    def generate_data(self, ix, n_digits=8, normalize=True, scale_factor=1.2):
        ''' Generate image with n_digits random MNIST digits om it
        Parameters
        ----------
        image : np.array
        '''
        # print('self.indices.shape[0] ', self.indices.shape[0])
        # self.predicted_bb = np.zeros((self.indices.shape[0], 1))
        # print('whats inside predicted bb', self.get(ix, 'predicted_bb'))
        
        self.predicted_bb = []
        image = self.get(ix, 'images')
        canvas = copy.deepcopy(self.get(ix, 'background'))
        # canvas = canvas / 255. because imresize will return data in [0, 255]
        canvas_size = canvas.shape[:2]
        # print(canvas_size, '  canvas_size')
        random_indices = np.random.choice(self.images.shape[0], n_digits)
        labels = np.array([self.labels[i] for i in random_indices]).reshape(-1)
        # scale_factor = np.random.uniform(low=0.7, high=1.5, size=(n_digits))
        scale_factor = np.repeat(np.random.uniform(low=1.5, high=1.9), n_digits)
        height, width = ((image.shape[0] * scale_factor).astype(np.int16), (image.shape[1] * scale_factor).astype(np.int16))
        random_images = [imresize(self.images[random_indices[i]], (height[i], width[i])) for i in range(random_indices.shape[0])]

        interval = np.random.randint(0, max(1e-5, (canvas_size[1] - np.max(width) * n_digits) / (2 * n_digits)))
        try:
            shift_right = np.random.randint(low=-interval, high=max(1, interval / 4), size=(n_digits))
            # shift_down = np.random.randint(low=-interval, high=max(1, interval / 4), size=(n_digits))
            shift_right = np.zeros(n_digits, np.int8)
            shift_down = np.zeros(n_digits, np.int8)
        except Exception as e:
            print('failed in shift', e)
        # print('interval', interval)
        # print('shift right', shift_right)
        # print('shift down', shift_down)

        coordinates = []
        confidence = np.zeros((n_digits, 1))
        # print('low=' , np.absolute(np.minimum(np.amin(shift_down), 0)))
        # print('high= ', canvas_size[0] - np.max(height) - np.max(np.absolute(shift_down)))
        try:
            left_x = np.random.randint(low=np.absolute(np.minimum(np.amin(shift_down), 0)), \
                high=canvas_size[0] - np.max(height) - np.max(np.absolute(shift_down)))
        except Exception as e:
            print('failed here ', e)
        try:
            # print('in low=' , np.absolute(np.minimum(np.amin(shift_right), 0)))
            # print('in high= ', max(1e-5, canvas_size[1] - (np.max(width) + interval + np.max(shift_right)) * n_digits))

            left_y = np.random.randint(low=np.absolute(np.minimum(np.amin(shift_right), 0)), high=max(1e-5, canvas_size[1] - (np.max(width) + interval + np.max(shift_right)) * n_digits))
        except Exception as e:
            print('in low=' , np.absolute(np.minimum(np.amin(shift_right), 0)))
            print('in high= ', max(1e-5, canvas_size[1] - (np.max(width) + interval + np.max(shift_right)) * n_digits))

            print(e, 'new_size ', canvas_size, ' (width + interval) * 8) ',  (np.max(width) + interval + np.max(shift_right)) * n_digits)
        right_y = 0
        gap_number = int(np.random.normal(loc=0, scale=1.2))
        for index, random_image in enumerate(random_images):
            random_image = 1 / 255. * random_image
            height = random_image.shape[0]
            width = random_image.shape[1]

            left_y = right_y + interval + shift_right[index]
            right_y = left_y + width
            shifted_left_x = left_x + shift_down[index]

            if gap_number > 0 and index <= gap_number or gap_number < 0 and index >= n_digits + gap_number:
                try:
                    confidence[index, 0] = 0
                except Exception as e:
                    print('flag2', e)

            else:
                try:
                    canvas[shifted_left_x:shifted_left_x + height, left_y:right_y, :] = random_image
                except Exception as e:
                    print('canvas crop in generate data', e, random_image.shape, canvas.shape)
                    print('failed trying to crop ', shifted_left_x, shifted_left_x + height, left_y, right_y)

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
        # print('inside generate_data: canvas.shape is', canvas.shape)
        return canvas, labels, coordinates, confidence




    @action
    @inbatch_parallel(init='indices', post='_assemble', components=('cropped_images', 'cropped_labels'))
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
        # print('reshaped predictions SHAPE ', predictions.shape)


        coordinates = predictions[:, :4]
        predicted_confidence = predictions[:, 4:5]
#         coordinates = predictions[:, 1:]
#         predicted_confidence = predictions[:, 0]


        left_corners = [coordinates[i][0] for i in range(n_digits)]
        sorted_indices = np.argsort(left_corners)
        coordinates = coordinates[sorted_indices]
        predicted_confidence = expit(predicted_confidence[sorted_indices])

        denormalized_coordinates = self.denormalize_bb(self.new_images.shape[1:3], coordinates)
        # print(denormalized_coordinates.shape, 'denormalized_coordinates')
        try:
            images = self.get(idx, 'new_images')
            labels = self.get(idx, 'labels')
        except Exception as e:
            print('get images', e)
        cropped_images = []
        cropped_labels = []
        for i in range(n_digits):
            # print('pred conf i', i , predicted_confidence[i])
            try:
                if predicted_confidence[i] < confidence_treshold:
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
                plt.imshow(current)
                plt.show()
            except Exception as e:
                print('SLICING ERROR', current_coords, e)

            try:
                cropped_images.append(imresize(current, new_size))
                cropped_labels.append(labels[i])
            except Exception as e:
                cropped_images.append(np.zeros((new_size[0], new_size[1], 3)))
                cropped_labels.append(0)
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
            return [], []
        # print(cropped_images.shape, 'CROPPED SHAPE')
        return cropped_images, cropped_labels

    def denormalize_bb(self, img_size, coordinates, n_digits=8):
        height, width = img_size
        # print('height width ', height, width)
        coordinates = copy.deepcopy(coordinates)
        coordinates = coordinates.reshape(-1, 4)
        max_boarders = np.ones((coordinates.shape[0]))
        min_boarders = np.zeros((coordinates.shape[0]))
        scales = [height, width, height, width]

        for i in range(4):
            coordinates[:, i] = np.minimum(coordinates[:, i], max_boarders)
            coordinates[:, i] = np.maximum(coordinates[:, i], min_boarders)
            coordinates[:, i] *= scales[i]
        coordinates[:, 2] += coordinates[:, 0]
        coordinates[:, 3] += coordinates[:, 1]
        # print(coordinates.shape)
        # print('coord ', coordinates)
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
        if isinstance(label, (list, tuple)):
            print('here')
            one_hot = np.zeros((len(label), 10))
            for i in range(len(label)):
                one_hot[i, label[i]] = 1
        else:
            one_hot = np.zeros(10)
            one_hot[label] = 1
        # print(one_hot.shape)
        return (one_hot.reshape(-1),)


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
