from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import skimage.io
from skimage import img_as_ubyte
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image

flags = tf.app.flags
conf = flags.FLAGS


class DataSet(object):
    def __init__(self):
        self.train_hr_img = []
        self.train_lr_img = []
        self.val_hr_img = []
        self.val_lr_img = []
        self.test_lr_img = []

    def load_dataset(self, hr_img_list, lr_img_list, val_hr_img_list, val_lr_img_list, sess, num_raw_train=11, num_raw_val=100, num_patch=1):
        # Load train images

        hr_all_images = np.ndarray([num_raw_train * num_patch, conf.im_size, conf.im_size, 1], dtype=np.uint8)
        # for i in range(num_raw_train):
        #     img = skimage.io.imread(hr_img_list[i], as_grey=True)
        #     patches = image.extract_patches_2d(img, [conf.im_size, conf.im_size], 50)  # The output patch shape is [50, 64, 64]
        #     patches = np.expand_dims(patches, axis=3)
        #     patches = img_as_ubyte(patches)
        #     hr_all_images[i * num_patch: i * num_patch + num_patch] = patches
        #     del img, patches
        #     print(i)

        for i in range(11):
            img = skimage.io.imread(hr_img_list[i], as_grey=True)
            img = np.expand_dims(img, axis=3)
            hr_all_images[i] = img
            del img
            print(i)

        idx_list = np.random.permutation(num_raw_train * num_patch)
        idx_list = idx_list[:conf.num_train_images]
        hr_images = hr_all_images[idx_list]
        del hr_all_images, idx_list

        # hr_images_tmp = np.ndarray([conf.num_train_images, conf.hr_size, conf.hr_size, 1], dtype=np.uint8)
        # for i in range(conf.num_train_images):
        #     image_pil = np.reshape(hr_images[i], [conf.im_size, conf.im_size])
        #     image_pil = Image.fromarray(image_pil.astype('uint8'))
        #     image_pil = image_pil.resize((conf.hr_size, conf.hr_size), Image.ANTIALIAS)
        #     image_pil = np.expand_dims(np.array(image_pil), axis=3)
        #     hr_images_tmp[i] = image_pil
        #
        # hr_images = hr_images_tmp
        # del hr_images_tmp

        lr_images = hr_images
        lr_images_tmp = np.ndarray([conf.num_train_images, conf.lr_size, conf.lr_size, 1], dtype=np.uint8)
        for i in range(conf.num_train_images):
            image_pil = np.reshape(lr_images[i], [conf.hr_size, conf.hr_size])
            image_pil = Image.fromarray(image_pil.astype('uint8'))
            image_pil = image_pil.resize((conf.lr_size, conf.lr_size), Image.ANTIALIAS)
            image_pil = np.expand_dims(np.array(image_pil), axis=3)
            lr_images_tmp[i] = image_pil

        lr_images = lr_images_tmp
        del lr_images_tmp

        hr_images = tf.cast(hr_images, tf.float32)
        lr_images = tf.cast(lr_images, tf.float32)
        hr_images = self.normalize_imgs_fn(hr_images)
        lr_images = self.normalize_imgs_fn(lr_images)

        # # This block is for loading the DIV2K validation data
        # # Load val hr images
        # val_hr_all_images = np.ndarray([num_raw_val * num_patch, conf.im_size, conf.im_size, 1], dtype=np.uint8)
        # for i in range(num_raw_val):
        #     img = skimage.io.imread(val_hr_img_list[i], as_grey=True)
        #     patches = image.extract_patches_2d(img, [conf.im_size, conf.im_size], 50)  # The output patch shape is [50, 64, 64]
        #     patches = np.expand_dims(patches, axis=3)
        #     patches = img_as_ubyte(patches)
        #     val_hr_all_images[i * num_patch: i * num_patch + num_patch] = patches
        #     del img, patches
        #     print(i)
        # idx_list = np.random.permutation(num_raw_val * num_patch)
        # idx_list = idx_list[:conf.num_val_images]
        # val_hr_images = val_hr_all_images[idx_list]
        # del val_hr_all_images, idx_list
        #
        # val_hr_images = tf.image.resize_images(val_hr_images, [conf.hr_size, conf.hr_size])
        # val_lr_images = tf.image.resize_images(val_hr_images, [conf.lr_size, conf.lr_size])
        # val_hr_images = tf.cast(val_hr_images, tf.float32)
        # val_lr_images = tf.cast(val_lr_images, tf.float32)
        # val_hr_images = self.normalize_imgs_fn(val_hr_images)
        # val_lr_images = self.normalize_imgs_fn(val_lr_images)

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # This block is for loading the 11 test image validation data
        # Load val hr images
        val_hr_all_images = np.ndarray([conf.num_val_images, conf.im_size, conf.im_size, 1], dtype=np.uint8)

        for i in range(11):
            img = skimage.io.imread(val_hr_img_list[i], as_grey=True)
            img = np.expand_dims(img, axis=3)
            val_hr_all_images[i] = img
            del img
            print(i)
        idx_list = np.random.permutation(11)
        idx_list = idx_list[:conf.num_val_images]
        val_hr_images = val_hr_all_images[idx_list]
        del val_hr_all_images, idx_list

        # val_hr_images_tmp = np.ndarray([conf.num_val_images, conf.hr_size, conf.hr_size, 1], dtype=np.uint8)
        # for i in range(conf.num_val_images):
        #     image_pil = np.reshape(val_hr_images[i], [conf.im_size, conf.im_size])
        #     image_pil = Image.fromarray(image_pil.astype('uint8'))
        #     image_pil = image_pil.resize((conf.hr_size, conf.hr_size), Image.ANTIALIAS)
        #     image_pil = np.expand_dims(np.array(image_pil), axis=3)
        #     val_hr_images_tmp[i] = image_pil
        #
        # val_hr_images = val_hr_images_tmp
        # del val_hr_images_tmp

        val_lr_images = val_hr_images
        val_lr_images_tmp = np.ndarray([conf.num_val_images, conf.lr_size, conf.lr_size, 1], dtype=np.uint8)
        for i in range(conf.num_val_images):
            image_pil = np.reshape(val_lr_images[i], [conf.hr_size, conf.hr_size])
            image_pil = Image.fromarray(image_pil.astype('uint8'))
            image_pil = image_pil.resize((conf.lr_size, conf.lr_size), Image.ANTIALIAS)
            image_pil = np.expand_dims(np.array(image_pil), axis=3)
            val_lr_images_tmp[i] = image_pil

        val_lr_images = val_lr_images_tmp
        del val_lr_images_tmp

        val_hr_images = tf.cast(val_hr_images, tf.float32)
        val_lr_images = tf.cast(val_lr_images, tf.float32)
        val_hr_images = self.normalize_imgs_fn(val_hr_images)
        val_lr_images = self.normalize_imgs_fn(val_lr_images)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        hr_images, lr_images, val_hr_images, val_lr_images = sess.run([hr_images, lr_images, val_hr_images, val_lr_images])
        print([hr_images.shape, lr_images.shape, val_hr_images.shape, val_lr_images.shape])

        return hr_images, lr_images, val_hr_images, val_lr_images

    def normalize_imgs_fn(self, x):
        x = x * (2. / 255.) - 1.
        # x = x * (1./255.)
        return x

    def load_file_list(self, train_hr_file, train_lr_file, valid_hr_file, valid_lr_file):
        train_hr_file_list = []
        train_lr_file_list = []
        valid_hr_file_list = []
        valid_lr_file_list = []

        for filename in [y for y in os.listdir(train_hr_file) if os.path.isfile(os.path.join(train_hr_file, y))]:
            train_hr_file_list.append("%s%s" % (train_hr_file, filename))

        for filename in [y for y in os.listdir(train_lr_file) if os.path.isfile(os.path.join(train_lr_file, y))]:
            train_lr_file_list.append("%s%s" % (train_lr_file, filename))

        for filename in [y for y in os.listdir(valid_hr_file) if os.path.isfile(os.path.join(valid_hr_file, y))]:
            valid_hr_file_list.append("%s%s" % (valid_hr_file, filename))

        for filename in [y for y in os.listdir(valid_lr_file) if os.path.isfile(os.path.join(valid_lr_file, y))]:
            valid_lr_file_list.append("%s%s" % (valid_lr_file, filename))

        return sorted(train_hr_file_list), sorted(train_lr_file_list), sorted(valid_hr_file_list), sorted(valid_lr_file_list)
