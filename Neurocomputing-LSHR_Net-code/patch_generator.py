#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Generate image patches
The code should be run in Anaconda since their Python library has sklearn.feature_extraction package.

@author: fangliang
"""

import glob
import os

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image

# Crop size
crop_size = (32, 32)

# Parameters
img_fmt = '.png'
in_path = '/home/fangliang/DIV2K/DIV2K_train_HR-original/*'
out_path = '/home/fangliang/DIV2K/DIV2K_train_HR-original-patches_val/'
directory = os.path.dirname(out_path)
try:
    os.stat(directory)
except:
    os.mkdir(directory)

# Read images
image_list = map(Image.open, glob.glob(in_path + img_fmt))

# Crop patches
for i in range(800, 850):  # todo: change the range
    pix = np.array(image_list[i])
    patches = image.extract_patches_2d(pix, crop_size, 10)

    # Save images
    for j in range(len(patches)):
        patch = Image.fromarray(patches[j])
        patch = patch.convert('L')
        patch.save(out_path + str(i) + '_' + str(j) + '.png', 'PNG')

        del patch

    del patches
