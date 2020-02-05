########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import pickle

import numpy as np
import tensorflow as tf

flags = tf.app.flags
conf = flags.FLAGS


########################################################################
class Cifar10(object):
    def __init__(self):
        # Directory where you want to download and save the data-set.
        # Set this before you start calling any of the functions below.
        self.data_path = conf.cifar10_path

        # URL for the data-set on the internet.
        self.data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        ########################################################################
        # Various constants for the size of the images.
        # Use these constants in your own program.

        # Width and height of each image.
        self.img_size = 32

        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self.num_channels = 3

        # Length of an image when flattened to a 1-dim array.
        self.img_size_flat = self.img_size * self.img_size * self.num_channels

        # Number of classes.
        self.num_classes = 10

        ########################################################################
        # Various constants used to allocate arrays of the correct size.

        # Number of files for the training-set.
        self._num_files_train = 5

        # Number of images for each batch-file in the training-set.
        self._images_per_file = 10000

        # Total number of images in the training-set.
        # This is used to pre-allocate arrays for efficiency.
        self._num_images_train = self._num_files_train * self._images_per_file

    def load_dataset(self, sess):
        # Train
        hr_all_images, _ = self.load_training_data()
        hr_images = hr_all_images[:conf.num_train_images]
        del hr_all_images

        hr_images = self.normalize_imgs_fn(hr_images)
        hr_images = tf.image.resize_images(hr_images, [conf.hr_size, conf.hr_size])
        lr_images = tf.image.resize_images(hr_images, [conf.lr_size, conf.lr_size])
        hr_images = tf.cast(hr_images, tf.float32)
        lr_images = tf.cast(lr_images, tf.float32)
        if conf.img_channel == 1:
            hr_images = hr_images[:, :, :, 1]
            lr_images = lr_images[:, :, :, 1]
            hr_images = tf.reshape(hr_images, shape=[conf.num_train_images, conf.hr_size, conf.hr_size, 1])
            lr_images = tf.reshape(lr_images, shape=[conf.num_train_images, conf.lr_size, conf.lr_size, 1])

        # Val
        val_hr_all_images, _ = self.load_test_data()
        val_hr_images = val_hr_all_images[:conf.num_val_images]
        del val_hr_all_images

        val_hr_images = self.normalize_imgs_fn(val_hr_images)
        val_hr_images = tf.image.resize_images(val_hr_images, [conf.hr_size, conf.hr_size])
        val_lr_images = tf.image.resize_images(val_hr_images, [conf.lr_size, conf.lr_size])
        val_hr_images = tf.cast(val_hr_images, tf.float32)
        val_lr_images = tf.cast(val_lr_images, tf.float32)
        if conf.img_channel == 1:
            val_hr_images = val_hr_images[:, :, :, 1]
            val_lr_images = val_lr_images[:, :, :, 1]
            val_hr_images = tf.reshape(val_hr_images, shape=[conf.num_val_images, conf.hr_size, conf.hr_size, 1])
            val_lr_images = tf.reshape(val_lr_images, shape=[conf.num_val_images, conf.lr_size, conf.lr_size, 1])

        hr_images, lr_images, val_hr_images, val_lr_images = sess.run(
            [hr_images, lr_images, val_hr_images, val_lr_images])

        return hr_images, lr_images, val_hr_images, val_lr_images

    ########################################################################
    # Private functions for downloading, unpacking and loading data-files.

    def _get_file_path(self, filename=""):
        """
        Return the full path of a data-file for the data-set.
        If filename=="" then return the directory of the files.
        """

        return os.path.join(self.data_path, "cifar-10-batches-py/", filename)

    def _unpickle(self, filename):
        """
        Unpickle the given file and return the data.
        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        file_path = self._get_file_path(filename)

        print("Loading data: " + file_path)

        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file)
            # data = pickle.load(file, encoding='bytes')

        return data

    def _convert_images(self, raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float)

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self.num_channels, self.img_size, self.img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images

    def _load_data(self, filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        cls = np.array(data[b'labels'])

        # Convert the images.
        images = self._convert_images(raw_images)

        return images, cls

    ########################################################################
    # Public functions that you may call to download the data-set from
    # the internet and load the data into memory.

    def load_training_data(self):
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[self._num_images_train, self.img_size, self.img_size, self.num_channels], dtype=float)
        cls = np.zeros(shape=[self._num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(self._num_files_train):
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self._load_data(filename="data_batch_" + str(i + 1))

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, cls

    def load_test_data(self):
        """
        Load all the test-data for the CIFAR-10 data-set.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        images, cls = self._load_data(filename="test_batch")

        return images, cls

    def normalize_imgs_fn(self, x):
        x = x * (2. / 255.) - 1.
        # x = x * (1./255.)
        return x
        ########################################################################
