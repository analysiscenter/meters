""" Custom class for detection CNN
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..//..")
from meters.dataset.models.tf.resnet import ResNet, ResNet18
from meters.dataset.dataset.models.tf import TFModel

from meters.dataset.dataset.models.tf.layers import conv_block




class NearestDetection(ResNet18):
    def build_config(self, names=None):
        config = super().build_config(names)
        num_digits = config.pop('num_digits')
        config['head']['num_digits'] = num_digits
        config['head']['units'] = 4 * num_digits
        return config

    def head(self, inputs, num_digits, name='head', **kwargs):
        kwargs = self.fill_params('head', **kwargs)
        outputs = conv_block(inputs, name=name, **kwargs)
        print('outputs.shape ', outputs.shape)

        all_coordinates = tf.get_default_graph().get_tensor_by_name("NearestDetection/inputs/coordinates:0")
        print('all_coordinates', all_coordinates.get_shape().as_list())
        # all_coordinates = self.coordinates
        # width = self.images.get_shape().as_list()[1]
        width = self.images.get_shape().as_list()[2]

        all_distances_list = []
        true_normalized = []
        for index in range(num_digits):
            current_normalized = all_coordinates[:, index, :]
            # = normalize_bbox(width, all_coordinates[:, index, :])
            print('current_normalized ', current_normalized.get_shape().as_list())
            true_normalized.append(current_normalized)
            print('outputs ', outputs.get_shape().as_list())
            all_dists = []
            for j in range(num_digits):
                all_dists.append(tf.reduce_mean(tf.square(current_normalized - outputs[:, 4 * j : 4 * (j + 1)]), axis=1))
            all_distances_list.append(tf.stack(all_dists, axis=1))

        true_normalized = tf.stack(true_normalized, axis=1)
        tf.identity(true_normalized, name='true_normalized')
        all_distances = tf.stack(all_distances_list, axis=2)
        print('all_distances ', all_distances.get_shape().as_list())

        min_distances = tf.reduce_min(all_distances, axis=1)

        print('min_distances ', min_distances.get_shape().as_list())
        tf.identity(min_distances, name='min_distances')
        
        average_min_distances = tf.reduce_mean(min_distances)
        print('min_distances after', average_min_distances.get_shape().as_list())
        tf.losses.add_loss(average_min_distances)

        predicted_bb = []
        arg_mins = tf.argmin(all_distances, axis=1)
        for index in range(num_digits):
            min_index = tf.cast(arg_mins[:, index][1], tf.int32)
            predicted_bb.append(outputs[:, 4 * min_index: 4 * (min_index + 1)])
        predicted_bb = tf.stack(predicted_bb, 1)
        tf.identity(predicted_bb, name='predicted_bb')
        # tf.identity(tf.constant(np.ones((200, 8, 4)), dtype=tf.int32), name='predicted_bb')
        print('predicted_bb shape', predicted_bb.shape)


        # images = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/images:0")
        # predicted_bb = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/predicted_bb:0")
        # boxes = tf.reshape(predicted_bb, [-1])
        # box_ind = tf.reshape(tf.stack([tf.range(tf.shape(self.inputs['images'])[0])] * num_digits, axis=1), [-1])
        # new_inputs = tf.image.crop_and_resize(self.inputs['images'], boxes, box_ind,
        #                                       tf.constant(size, dtype=tf.int32))

        # labels = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/labels:0")
        # reshaped_labels = tf.reshape(labels, [-1])
        # tf.identity(reshaped_labels, name='reshaped_labels')

        # confidence = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/confidence:0")
        # reshaped_confidence = tf.reshape(confidence, [-1])
        # tf.identity(reshaped_confidence, name='reshaped_confidence')


        # current_huber = tf.losses.huber_loss(current_normalized, current_predictions)
        # tf.losses.add_loss(current_huber)

        tf.identity(min_distances, name='min_distances')


        # tf.identity(outputs[:, :], name='predicted_bb')
        tf.zeros([1], tf.int32, name='accuracy')
        tf.zeros([1], tf.int32, name='ce_loss')
        tf.zeros([1], tf.int32, name='mse_loss_history')
        tf.zeros([1], tf.int32, name='labels_hat')
        tf.zeros([1], tf.int32, name='all_acuracies')
        # print(tf.get_default_graph().get_operations())
        return outputs


class ClassificationModel(ResNet18):
    def build_config(self, names=None):
        config = super().build_config(names)
        num_digits = config.pop('num_digits')
        config['head']['num_digits'] = num_digits
        config['input_block']['num_digits'] = num_digits

        config['head']['units'] = 10
        return config

    def input_block(self, inputs, name='input_block', num_digits=8, size=(32, 16), **kwargs):
        """ Transform inputs with a convolution block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor
        """
        images = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/images:0")
        predicted_bb = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/predicted_bb:0")
        boxes = tf.reshape(predicted_bb, [-1])
        box_ind = tf.reshape(tf.stack([tf.range(tf.shape(self.inputs['images'])[0])] * num_digits, axis=1), [-1])
        new_inputs = tf.image.crop_and_resize(self.inputs['images'], boxes, box_ind,
                                              tf.constant(size, dtype=tf.int32))

        labels = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/labels:0")
        reshaped_labels = tf.reshape(labels, [-1])
        tf.identity(reshaped_labels, name='reshaped_labels')

        confidence = tf.get_default_graph().get_tensor_by_name("ClassificationModel/inputs/confidence:0")
        reshaped_confidence = tf.reshape(confidence, [-1])
        tf.identity(reshaped_confidence, name='reshaped_confidence')
        
        kwargs = cls.fill_params('input_block', **kwargs)
        if kwargs.get('layout'):
            return conv_block(new_inputs, name=name, **kwargs)
        return new_inputs


    def head(self, inputs, num_digits, name='head', **kwargs):
        kwargs = self.fill_params('head', **kwargs)
        outputs = conv_block(inputs, name=name, **kwargs)
        print('outputs.shape ', outputs.shape)

