#!/usr/bin/python

import argparse
import time
import json

import numpy as np
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf
import cv2

from utils import apply_color_map
import model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint')
parser.add_argument('--test_image', type=str, default='output/test1.JPG', help='path to input test image')
opt = parser.parse_args()

print(opt)

#### Test ####

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

net = model.build_bn(480, 320, 3, train=True)

# Model
if opt.checkpoint:
    net.load_weights(opt.checkpoint, by_name=True)
else:
    print('No checkpoint specified! Set it with the --checkpoint argument option')
    exit()


# Testing
image_height = net.inputs[0].shape[2]
image_width = net.inputs[0].shape[1]
print(image_width,image_height)
x = np.array([cv2.resize(cv2.imread(opt.test_image, 1), (image_height, image_width))])

times = []
for i in range(100):
	start_time = time.time()

	with tf.device('/gpu:0'):
	    y = net.predict(np.array(x), batch_size=1)

	duration = time.time() - start_time
	times.append(duration)
print('Average = {}'.format(sum(times)/100))
print(times)

print('Generated segmentations in %s seconds -- %s FPS' % (duration, 1.0/duration))

# Save output image
with open('config.json') as config_file:
    config = json.load(config_file)
labels = config['labels']

output = apply_color_map(np.argmax(y[0], axis=-1), labels)


output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

print(x[0].shape)
# print(y[0].shape)
print(cv2.resize(output, (480, 320)).shape)
# x = x[0]

im = x[0]
overlay = cv2.resize(output, (480, 320))
alpha = 0.5
cv2.addWeighted(overlay, alpha, im, 1 - alpha,0, im)
cv2.imshow("Output", im)
cv2.waitKey(0)

# cv2.imwrite('output_sample.png', cv2.resize(output, (image_width, image_width)))
# cv2.imwrite('input_sample.png', cv2.resize(cv2.imread(opt.test_image, 1), (image_width, image_width)))

###############
