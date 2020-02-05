from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import tensorlayer as tl
from tensorlayer.layers import *
from visualize_tensor import *
import numpy as np

flags = tf.app.flags
conf = flags.FLAGS


def resnet_block(inputs, channels=64, kernel_size=3):
    tmp = slim.conv2d(inputs, num_outputs=64, kernel_size=11, activation_fn=None)
    tmp = tf.layers.batch_normalization(tmp)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, num_outputs=32, kernel_size=1, activation_fn=None)
    tmp = tf.layers.batch_normalization(tmp)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(inputs, num_outputs=1, kernel_size=7, activation_fn=None)
    tmp = tf.add(inputs, tmp)
    return tmp


def upsample(inputs, scale=2, activation=tf.nn.relu):
    assert scale in [2, 3, 4]
    if scale == 2:
        ps_features = conf.img_channel * (scale ** 2)
        x = slim.conv2d(inputs, ps_features, [3, 3], activation_fn=activation)
        if conf.img_channel == 1:
            x = PS(x, 2, color=False)
        else:
            x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3 * (scale ** 2)
        inputs = slim.conv2d(inputs, ps_features, [3, 3], activation_fn=activation)
        # inputs = slim.conv2d_transpose(inputs,ps_features,9,stride=1,activation_fn=activation)
        inputs = PS(inputs, 3, color=True)
    elif scale == 4:
        ps_features = 3 * (2 ** 2)
        for i in range(2):
            inputs = slim.conv2d(inputs, ps_features, [3, 3], activation_fn=activation)
            # inputs = slim.conv2d_transpose(inputs,ps_features,6,stride=1,activation_fn=activation)
            inputs = PS(inputs, 2, color=True)
    return inputs


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def ResidualSingleLevel(net_image, net_feature, num_features=64, reuse=False):
    with tf.variable_scope("Model_level", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_tmp = net_feature
        # recursive block
        for i in range(6):
            net_tmp = residual_unit(net_tmp, num_features, i)
            net_tmp = ElementwiseLayer(layers=[net_feature, net_tmp], combine_fn=tf.add, name='add_feature')

        net_feature = net_tmp
        net_feature = PReluLayer(net_feature, name='prelu_feature', a_init=tf.constant_initializer(value=0.2))
        # net_feature = Conv2dLayer(net_feature, shape=(3, 3, num_features, num_features*4), strides=(1, 1, 1, 1),
        #                           name='upconv_feature', W_init=tf.contrib.layers.xavier_initializer(), b_init=None)
        # net_feature = SubpixelConv2d(net_feature, scale=2, n_out_channel=num_features, name='subpixel_feature')

        # net_feature = DeConv2dLayer(net_feature, act=tf.identity, shape=(4, 4, 64, 64), strides=(1, 2, 2, 1),
        #                             output_shape=(conf.batch_size, conf.lr_size * 2, conf.lr_size * 2, 64), padding='SAME',
        #                             W_init=None, b_init=None, name='devcon1',
        #                             W_init_args=get_bilinear_filter([4, 4, 64, 64], 2))
        weights_f = get_bilinear_filter([4, 4, 64, 64], 2, name='weights_f')
        net_feature = tf.nn.conv2d_transpose(net_feature.outputs, weights_f,
                                             output_shape=[conf.batch_size, conf.lr_size * 2, conf.lr_size * 2, 64],
                                             strides=(1, 2, 2, 1), padding='SAME')
        net_feature = InputLayer(net_feature)

        # add image back
        gradient_level = Conv2dLayer(net_feature, shape=(3, 3, num_features, conf.img_channel), strides=(1, 1, 1, 1),
                                     act=leaky_relu, W_init=tf.contrib.layers.variance_scaling_initializer(),
                                     b_init=None, name='gradient_level')

        # net_image = Conv2dLayer(net_image, shape=(3, 3, conf.img_channel, 4), strides=(1, 1, 1, 1),
        #                         name='upconv_image', W_init=tf.contrib.layers.xavier_initializer(), b_init=None)
        # net_image = SubpixelConv2d(net_image, scale=2, n_out_channel=1, name='subpixel_image')
        # net_image = DeConv2dLayer(net_image, act=tf.identity, shape=(4, 4, 1, 64), strides=(1, 2, 2, 1),
        #                           output_shape=(conf.batch_size, conf.lr_size * 2, conf.lr_size * 2, 1), padding='SAME',
        #                           W_init=None, b_init=None, name='devcon2',
        #                           W_init_args=get_bilinear_filter([4, 4, 64, 1], 2))
        weights_im = get_bilinear_filter([4, 4, 1, 1], 2, 'weights_im')
        net_image = tf.nn.conv2d_transpose(net_image.outputs, weights_im,
                                           output_shape=[conf.batch_size, conf.lr_size * 2, conf.lr_size * 2, 1],
                                           strides=(1, 2, 2, 1), padding='SAME')
        net_image = InputLayer(net_image)
        net_image = ElementwiseLayer(layers=[gradient_level, net_image], combine_fn=tf.add, name='add_image')

    return net_image, net_feature, gradient_level


def leaky_relu(x, leaky_alpha=0.2):
    return tf.maximum(x, leaky_alpha * x)


def residual_unit(net_tmp, num_features, ind):
    net_tmp = PReluLayer(net_tmp,
                         name='prelu_1_D%s' % ind, a_init=tf.constant_initializer(value=0.2))
    net_tmp = Conv2dLayer(net_tmp,
                          shape=(3, 3, num_features, num_features),
                          strides=(1, 1, 1, 1),
                          name='conv_1_D%s' % ind,
                          W_init=tf.contrib.layers.variance_scaling_initializer(), b_init=None)
    net_tmp = PReluLayer(net_tmp,
                         name='prelu_2_D%s' % ind, a_init=tf.constant_initializer(value=0.2))
    net_tmp = Conv2dLayer(net_tmp,
                          shape=(3, 3, num_features, num_features),
                          strides=(1, 1, 1, 1),
                          name='conv_2_D%s' % ind,
                          W_init=tf.contrib.layers.variance_scaling_initializer(), b_init=None)
    return net_tmp


def get_bilinear_filter(filter_shape, upscale_factor, name):
    # filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    # Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            # Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name=name, initializer=init,
                                       shape=weights.shape)
    return bilinear_weights
# ============================ Binary-Neural-Network ============================


def hard_sigmoid(x):
    return tf.contrib.keras.backend.clip((x + 1.0) / 2.0, 0.0, 1.0)


def binarization(W, H=1, binary=False, stochastic=False):
    if not binary:
        Wb = W
    else:
        Wb = hard_sigmoid(W / H)
        if stochastic:
            # use hard sigmoid weight for possibility
            Wb = tf.contrib.keras.backend.random_binomial(tf.shape(Wb), p=Wb)
        else:
            # round weight to 0 and 1
            Wb = tf.round(Wb)
        # change range from 0~1  to  -1~1
        Wb = Wb * 2 - 1
    return Wb


class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, padding="SAME", binary=False, stochastic=False,
                 is_training=None, index=0):
        # binary: whether to implement the Binary Connect
        # stochastic: whether implement stochastic weight if do Binary Connect

        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                # the real value of weight
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer())
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.constant_initializer(0.))
                self.bias = bias

            # use real value weights to test in stochastic BinaryConnect
            if binary and stochastic:
                wb = tf.cond(is_training, lambda: binarization(weight, H=1, binary=binary, stochastic=stochastic),
                             lambda: weight)
            # otherwise, return binarization directly
            else:
                wb = binarization(weight, H=1, binary=binary, stochastic=stochastic)

            self.wb = wb

            cell_out = tf.nn.conv2d(input_x, self.wb, strides=[1, 1, 1, 1], padding=padding)

            cell_out = tf.add(cell_out, bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), self.weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), self.bias)

            # to store the moments for adam
            with tf.name_scope('conv_moment'):
                self.m_w = tf.get_variable(name='conv_first_moment_w_%d' % index, shape=w_shape,
                                           initializer=tf.constant_initializer(0.))
                self.v_w = tf.get_variable(name='conv_second_moment_w_%d' % index, shape=w_shape,
                                           initializer=tf.constant_initializer(0.))
                self.m_b = tf.get_variable(name='conv_first_moment_b_%d' % index, shape=b_shape,
                                           initializer=tf.constant_initializer(0.))
                self.v_b = tf.get_variable(name='conv_second_moment_b_%d' % index, shape=b_shape,
                                           initializer=tf.constant_initializer(0.))

    def output(self):
        return self.cell_out


# ============================ tensorpack ============================

def quantize(x, k):
    n = float(2 ** k - 1)
    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x * n) / n


def fw(x):
    G = tf.get_default_graph()

    with G.gradient_override_map({"Sign": "Identity"}):
        E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
        return tf.sign(x / E) * E
    x = tf.tanh(x)
    x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
    return 2 * quantize(x, bitW) - 1


def remap_variables(fn):
    """
    Use fn to map the output of any variable getter.

    Args:
        fn (tf.Variable -> tf.Tensor)

    Returns:
        a context where all the variables will be mapped by fn.

    Example:
        .. code-block:: python

            with varreplace.remap_variables(lambda var: quantize(var)):
                x = FullyConnected('fc', x, 1000)   # fc/{W,b} will be quantized
    """

    def custom_getter(getter, *args, **kwargs):
        v = getter(*args, **kwargs)
        return fn(v)

    return custom_getter_scope(custom_getter)


def custom_getter(getter, *args, **kwargs):
    v = getter(*args, **kwargs)
    return fw(v)


# ============================ BinaryNet.tf ============================


def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Ceil": "Identity"}):
            x = tf.clip_by_value(x, -0.8, 0.8)
            return tf.ceil(x)


def bc_conv2d(x, nOutputPlane, kW, kH, dW=1, dH=1, padding='VALID', bias=False, reuse=None,
              name='BinarizedWeightOnlySpatialConvolution', is_training=True):
    nInputPlane = x.get_shape().as_list()[3]
    with tf.variable_op_scope([x], None, name, reuse=reuse):
        # Random weights initialisation
        w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # # Bernoulli weights initialisation
        # bernoulli = tf.distributions.Bernoulli(probs=0.5)
        # matrix = bernoulli.sample([1, nOutputPlane, kH, kW])
        # matrix = tf.cast(matrix, tf.float32)
        # matrix = tf.transpose(matrix, [2, 3, 0, 1])
        # w = tf.Variable(matrix, trainable=True, name='mask')

        bin_w = binarize(w)
        out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)

        # Visualise weights in TensorBoard
        grid = put_kernels_on_grid(bin_w)
        tf.summary.image('linear_mapping/binary_kernels', grid, max_outputs=3)
        tf.summary.histogram("linear_mapping/binary_kernels", bin_w)

        if bias:
            b = tf.get_variable('bias', [nOutputPlane], initializer=tf.zeros_initializer)
            out = tf.nn.bias_add(out, b)
        return out
