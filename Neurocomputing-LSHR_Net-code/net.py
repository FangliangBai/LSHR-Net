from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ops import *
from visualize_tensor import *
import numpy as np

flags = tf.app.flags
conf = flags.FLAGS


class Net(object):
    def __init__(self, hr_img, lr_img, mask_type, is_linear_only, scope=None):
        # Initialize all parameters here
        self.hr_images = hr_img
        self.lr_images = lr_img
        with tf.variable_scope(scope) as scope:
            self.construct_net(hr_img, lr_img, mask_type, is_linear_only)

    def linear_mapping_network(self, inputs, mask_type):
        measurements = []

        with tf.variable_scope('linear_mapping') as scope:
            # Sample
            if mask_type == 'trained':
                measurements = tf.contrib.layers.conv2d(inputs,
                                                        num_outputs=conf.num_measure,
                                                        kernel_size=conf.lr_size,
                                                        padding='VALID',
                                                        activation_fn=None,
                                                        biases_initializer=None,
                                                        scope="sampling")
            elif mask_type == 'Bernoulli':
                bernoulli = tf.distributions.Bernoulli(probs=0.5)
                matrix = bernoulli.sample([conf.img_channel,
                                           conf.num_measure,
                                           16,
                                           16])  # [image_channel, num_mask, mask_height, mask_width]
                matrix = tf.cast(matrix, tf.float32)
                matrix = tf.transpose(matrix, [2, 3, 0, 1])
                mask = tf.Variable(matrix, trainable=False, name='mask')
                measurements = tf.nn.conv2d(inputs, mask, strides=[1, 8, 8, 1], padding="SAME")
            elif mask_type == 'trained_binary':
                measurements = bc_conv2d(inputs, nOutputPlane=conf.num_measure,
                                         kW=conf.lr_size, kH=conf.lr_size,
                                         dW=1, dH=1, padding='VALID', bias=True,
                                         reuse=None, is_training=False, name='BinarizedWeightOnlySpatialConvolution')
            elif mask_type == 'dropout':
                # Method 1_1
                # mask = tf.get_variable('mask', shape=[1, 1, conf.img_channel, conf.num_measure], dtype=tf.float32,
                #                        initializer=tf.ones_initializer(), trainable=False)
                # measurements = tf.nn.conv2d(inputs, mask, strides=[1, 1, 1, 1], padding="VALID")
                # measurements = tf.nn.dropout(measurements, keep_prob=conf.dropout, noise_shape=[1, conf.hr_size, conf.hr_size, conf.num_measure])
                # measurements = tf.multiply(measurements, conf.dropout)

                # # Method 2
                # mask = tf.get_variable('mask', shape=[3, 3, conf.img_channel, conf.num_measure], dtype=tf.float32,
                #                        initializer=tf.ones_initializer(), trainable=False)
                # measurements = tf.nn.conv2d(inputs, mask, strides=[1, 1, 1, 1], padding="VALID")
                # measurements = tf.nn.dropout(measurements, keep_prob=conf.dropout, noise_shape=[1, 14, 14, conf.num_measure])
                # measurements = tf.multiply(measurements, conf.dropout)

                # # Method 3
                # mask = tf.get_variable('mask', shape=[1, 1, conf.img_channel, conf.num_measure], dtype=tf.float32,
                #                        initializer=tf.ones_initializer(), trainable=False)
                # measurements = tf.nn.conv2d(inputs, mask, strides=[1, 1, 1, 1], padding="VALID")
                # measurements = tf.nn.dropout(measurements, keep_prob=conf.dropout,
                #                              noise_shape=[1, conf.hr_size, conf.hr_size, conf.num_measure])
                # measurements = tf.multiply(measurements, conf.dropout)
                # measurements = tf.reduce_sum(measurements,axis=[1,2],keep_dims=True)

                # Method 1_2
                mask = tf.get_variable('mask', shape=[8, 8, conf.img_channel, conf.num_measure], dtype=tf.float32,
                                       initializer=tf.ones_initializer(), trainable=False)
                measurements = tf.nn.conv2d(inputs, mask, strides=[1, 8, 8, 1], padding="VALID")
                measurements = tf.nn.dropout(measurements, keep_prob=conf.dropout, noise_shape=[1, 8, 8, conf.num_measure])
                measurements = tf.multiply(measurements, conf.dropout)

            # Reconstruct
            if mask_type == 'dropout':
                # Method 1
                input0 = slim.conv2d(measurements,
                                     num_outputs=256,
                                     kernel_size=1,
                                     stride=1,
                                     padding='SAME',
                                     weights_regularizer=tf.contrib.layers.l1_regularizer(scale=1e-3),
                                     biases_initializer=tf.zeros_initializer(),
                                     activation_fn=None)
                input0 = slim.maxout(input0, 256)
                input1 = slim.conv2d_transpose(input0,
                                               num_outputs=256,
                                               kernel_size=[16, 16],
                                               stride=8,
                                               padding='SAME',
                                               activation_fn=None)
                # input2 = slim.conv2d_transpose(input0,
                #                               num_outputs=256,
                #                               kernel_size=[8, 8],
                #                               stride=1,
                #                               padding='SAME',
                #                               activation_fn=None)
                # input3 = slim.conv2d_transpose(input0,
                #                                num_outputs=256,
                #                                kernel_size=[4, 4],
                #                                stride=1,
                #                                padding='SAME',
                #                                activation_fn=None)
                # input = tf.concat(values=[input1,input2,input3], axis=-1)
                # input = slim.maxout(input,128)
                input = slim.conv2d(input1,
                                    num_outputs=128,
                                    kernel_size=1,
                                    stride=2,
                                    padding='SAME',
                                    weights_regularizer=tf.contrib.layers.l1_regularizer(scale=1e-3),
                                    biases_initializer=tf.zeros_initializer(),
                                    activation_fn=None)
                input = slim.maxout(input, 64)
                input = slim.conv2d(input,
                                    num_outputs=64,
                                    kernel_size=1,
                                    stride=2,
                                    padding='SAME',
                                    weights_regularizer=tf.contrib.layers.l1_regularizer(scale=1e-3),
                                    biases_initializer=tf.zeros_initializer(),
                                    activation_fn=None)
                input = slim.maxout(input, 1)
                input = slim.conv2d(input,
                                    num_outputs=conf.img_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding='SAME',
                                    weights_regularizer=tf.contrib.layers.l1_regularizer(scale=1e-3),
                                    biases_initializer=tf.zeros_initializer(),
                                    activation_fn=None)  # Set act_func to None because the pixel value is in range [-1, 1]
                linear_mapping_logits = input
                tf.summary.image('linear_mapping/linear_outputs', linear_mapping_logits, max_outputs=3)

                return linear_mapping_logits

                # # Method 2
                # with slim.arg_scope([slim.conv2d_transpose],
                #                     biases_initializer=tf.constant_initializer(0),
                #                     activation_fn=None,
                #                     weights_regularizer=slim.l2_regularizer(0.01)):
                #     input = slim.conv2d_transpose(measurements, 128, kernel_size=[3, 3], stride=1, padding='VALID')
                #     input = slim.conv2d(input, 64, 1, 1, 'SAME', activation_fn=None)
                #     input = slim.conv2d(input, 32, 5, 1, 'SAME', activation_fn=None)
                #     input = slim.conv2d(input, 1, 1, 1, 'SAME', activation_fn=None)
                #     linear_mapping_logits = input

                # # Method 3
                # with slim.arg_scope([slim.fully_connected],
                #                     biases_initializer=tf.constant_initializer(0),
                #                     activation_fn=None,
                #                     weights_regularizer=slim.l2_regularizer(0.01)):
                #     input = slim.fully_connected(measurements, 256)
                #     input = tf.contrib.layers.maxout(input,128)
                #     input = slim.fully_connected(input, 256)
                #     input = tf.contrib.layers.maxout(input, 128)
                #     input = slim.fully_connected(input, 256)
                #     input = tf.transpose(input, [0, 2, 3, 1])
                #     linear_mapping_logits = tf.reshape(input, [conf.batch_size, conf.hr_size, conf.hr_size, conf.img_channel])
                #     return linear_mapping_logits

            else:
                # input = slim.conv2d(measurements,
                #                     num_outputs=conf.hr_size * conf.hr_size * conf.img_channel,
                #                     kernel_size=1,
                #                     stride=1,
                #                     padding='SAME',
                #                     weights_regularizer=None,
                #                     biases_initializer=tf.zeros_initializer(),
                #                     activation_fn=None)
                # input = tf.transpose(input, [0, 2, 3, 1])
                # linear_mapping_logits = tf.reshape(input, [conf.batch_size, conf.hr_size, conf.hr_size, conf.img_channel])
                # return linear_mapping_logits

                # with slim.arg_scope([slim.fully_connected],
                #                     biases_initializer=tf.constant_initializer(0),
                #                     activation_fn=None,
                #                     weights_regularizer=slim.l2_regularizer(0.01)):
                #     input = slim.fully_connected(measurements, conf.hr_size*conf.hr_size)
                #     input = tf.contrib.layers.maxout(input,int(conf.hr_size*conf.hr_size/2))
                #     input = slim.fully_connected(input, conf.hr_size*conf.hr_size)
                #     input = tf.contrib.layers.maxout(input, int(conf.hr_size*conf.hr_size/2))
                #     input = slim.fully_connected(input, conf.hr_size*conf.hr_size)
                #     input = tf.transpose(input, [0, 2, 3, 1])
                #     linear_mapping_logits = tf.reshape(input, [conf.batch_size, conf.hr_size, conf.hr_size, conf.img_channel])
                #     return linear_mapping_logits

                with slim.arg_scope([slim.conv2d_transpose],
                                    biases_initializer=tf.constant_initializer(0),
                                    activation_fn=None,
                                    weights_regularizer=slim.l2_regularizer(0.01)):
                    input = slim.conv2d_transpose(measurements, 1, kernel_size=[16, 16], stride=8, padding='SAME')
                    linear_mapping_logits = input
                    return linear_mapping_logits

    def residual_reducing_network(self, inputs, num_features=64):
        with tf.variable_scope('residual_blocks') as scope:
            inputs_level = InputLayer(inputs, name='input_level')

            net_feature = Conv2dLayer(inputs_level,
                                      shape=(3, 3, conf.img_channel, num_features),
                                      strides=(1, 1, 1, 1),
                                      W_init=tf.contrib.layers.variance_scaling_initializer(),
                                      b_init=None,
                                      name='init_conv')

            net_image = inputs_level

            # 2X for each level
            net_image1, net_feature1, net_gradient1 = LapSRNSingleLevel(net_image, net_feature, reuse=tf.AUTO_REUSE)
            net_image2, net_feature2, net_gradient2 = LapSRNSingleLevel(net_image1, net_feature1, reuse=True)

            return net_image2, net_gradient2, net_image1, net_gradient1

    def construct_net(self, hr_img, lr_img, mask_type, is_linear_only):
        self.linear_mapping_logits = self.linear_mapping_network(lr_img, mask_type)

        self.net_image2, self.net_gradient2, self.net_image1, self.net_gradient1 = self.residual_reducing_network(self.linear_mapping_logits)

        hr_img_down = tf.image.resize_images(hr_img, size=[conf.lr_size * 2, conf.lr_size * 2])

        # collect the regularization loss
        reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'linear_mapping')

        if is_linear_only:
            if conf.charbonnier_loss:
                self.loss = self.compute_charbonnier_loss(hr_img, self.linear_mapping_logits, is_mean=True) + tf.reduce_sum(reg_ws)
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.l2_loss(tf.losses.absolute_difference(hr_img, self.linear_mapping_logits))) + tf.reduce_sum(reg_ws)
            tf.summary.scalar('linear_loss_train', self.loss)
        else:
            if conf.charbonnier_loss:
                loss1 = self.compute_charbonnier_loss(self.net_image1.outputs, hr_img_down, is_mean=True)
                loss2 = self.compute_charbonnier_loss(self.net_image1.outputs, hr_img, is_mean=True)
                self.loss = loss2
                tf.summary.scalar('residual_loss_train', self.loss)
            else:
                loss1 = tf.reduce_mean(tf.nn.l2_loss(tf.losses.absolute_difference(hr_img, self.net_image1)))
                loss2 = tf.reduce_mean(tf.nn.l2_loss(tf.losses.absolute_difference(hr_img, self.net_image2)))
                self.loss =loss1 + loss2
            tf.summary.scalar('residual_loss_train', self.loss)

    def compute_charbonnier_loss(self, tensor1, tensor2, is_mean=True):
        epsilon = 1e-3
        if is_mean:
            loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1, tensor2)) + epsilon), [1, 2, 3]))
        else:
            loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1, tensor2)) + epsilon), [1, 2, 3]))
        return loss


''' Backup Zone
    This zone backups the experiment residual blocks during experiments.
'''


def residual_reducing_network(self, inputs, res_num=1, num_features=64, scale=2):
    with tf.variable_scope('residual_blocks') as scope:
        conv_1 = slim.conv2d(inputs, num_features, [3, 3])

        for i in range(res_num):
            inputs = resnet_block(inputs, num_features)

        inputs = slim.conv2d(inputs, num_features, [3, 3])
        # Up-sample output of the convolution
        inputs = upsample(inputs, scale, activation=None)
        # net_image1, net_feature1, net_gradient1 = LapSRNSingleLevel(inputs, conv_1, reuse=None)
        # inputs = slim.conv2d(inputs, 3, [3, 3])
        # One final convolution on the up-sampling output
        # inputs = inputs  # slim.conv2d(x,output_channels,[3,3])
        residual_reducing_logits = inputs

        return residual_reducing_logits
