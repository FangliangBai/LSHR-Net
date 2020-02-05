# import tensorflow as tf

# dropout = tf.placeholder(tf.float32)
# inputs = tf.Variable(tf.ones([3, 4, 4, 1]))
# # y = tf.nn.dropout(inputs, dropout, noise_shape=[1,4,4,1])
# mask = tf.get_variable('mask', shape=[1, 1, 1, 2], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)
# measurements = tf.nn.conv2d(inputs, mask, strides=[1, 1, 1, 1], padding="VALID")
# measurements = tf.nn.dropout(measurements, keep_prob=0.5, noise_shape=[1, 4, 4, 2])
# measurements = tf.multiply(measurements, 0.5)
# measurements = tf.reduce_sum(measurements, axis=[1, 2], keep_dims=True)
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# result = sess.run(measurements)
# print(result.shape)
# print(result[0,:,:,0])
# print('-------------')
# print(result[0,:,:,1])
# print('=============')
# print(result[1,:,:,0])
# print('-------------')
# print(result[1,:,:,1])
# print('=============')
# print(result[2,:,:,0])
# print('-------------')
# print(result[2,:,:,1])

# inputs = tf.Variable(tf.ones([3, 4, 4, 2]))
# measurements = tf.reduce_sum(inputs, axis=[1, 2], keepdims=True)
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# result = sess.run(measurements)
# print(result)


# bernoulli = tf.distributions.Bernoulli(probs=1.0)
# matrix = bernoulli.sample([1, 3, 4, 4])  # [image_channel, num_mask, mask_height, mask_width]
# with tf.variable_scope('residual_blocks') as scope:
#     mask1 = tf.Variable(matrix, trainable=False, name='mask')
# with tf.variable_scope('residual_blocks',reuse=True) as scope2:
#     mask2 = tf.get_variable('mask')
# print(mask1.name)
# print(mask2.name)
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# result = sess.run(mask2)
# print(result.dtype)
# print(result[0,0,:,:])
# print('-------------')
# print(result[0,0,:,:])
# print('=============')
# print(result[0,1,:,:])
# print('-------------')
# print(result[0,1,:,:])
# print('=============')
# print(result[0,2,:,:])
# print('-------------')
# print(result[0,2,:,:])


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt;

# with tf.variable_scope('V1', reuse=False):
#     a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
#
# with tf.variable_scope('V1', reuse=tf.AUTO_REUSE):
#     a3 = tf.get_variable('a1')
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(a1.name)
#     print(sess.run(a1))
#     print(a3.name)
#     print(sess.run(a3))

# import glob
# import os
# import skimage.io
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from sklearn.feature_extraction import image
# from skimage import img_as_ubyte
#
# train_hr_file_list = []

# directory = '/home/fangliang/DIV2K/DIV2K_train_HR-original/'
# for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory, y))]:
#     train_hr_file_list.append("%s%s" % (directory, filename))
#
# train_hr_file_list = sorted(train_hr_file_list)
#
# hr_images = np.ndarray([45000,64,64,1], dtype=np.uint8)
# for i in range(1, 901, 1):
#     img = skimage.io.imread(train_hr_file_list[i], as_grey=True)
#     patches = image.extract_patches_2d(img, [64, 64], 50)
#     patches = np.expand_dims(patches, axis=3)
#     patches = img_as_ubyte(patches)
#
#     hr_images[(i-1)*50:(i-1)*50+50] = patches
#     del img, patches
#     print(i)
#     del hr_images

# a = np.random.permutation(10)
# b = np.random.permutation(10)
# b = b[:4]
# print(a)
# print(b)
# print(a[b])

# import csv
# # filename = 'aaa.csv'
# # f = open(filename, 'a')
# # writer = csv.writer(f)
# # f.close()
# for row in range(10):
#     writeFileObj = open('result.csv', 'a')
#     writer = csv.writer(writeFileObj)
#     writer.writerow([1.0, 2.0])
#     writeFileObj.close()


import tensorflow as tf


for event in tf.train.summary_iterator('/media/kent/DISK2/sr_spc/models/events.out.tfevents.1526038863.kent-System-Product-Name'):
    for value in event.summary.value:
        print(value.tag)
        if value.HasField('simple_value'):
            print(value.simple_value)
