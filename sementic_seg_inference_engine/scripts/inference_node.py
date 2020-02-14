#!/usr/bin/env python

import time
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import numpy as np
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf
import cv2

from utils import apply_color_map
import model


class seg_node:

	def __init__(self):
		self.weights = "weights46.h5"
		self.net = model.build_bn(480, 320, 3, train=True)
		self.net.load_weights(self.weights, by_name=True)
		self.img_height = self.net.input[0].shape[2]
		self.img_width = self.net.inputs[0].shape[1]

		# Publisher
		self.seg_img_publisher = rospy.Publisher("seg_img",Image)
		# sensor_msgs/Image to cv_img
		self.bridge = CvBridge() 
		# Subscriber
		self.img_subscriber = rospy.Subscriber("/pointgrey_cam/image_raw", Image, self.inference_callback)


	def inference_callback(self, data):
		try:
			cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		x = np.array([cv2.resize(cv_img,(self.img_height, self.img_width))])
		y = self.net.predict(np.array(x), batch_size=1)
		output = apply_color_map(np.argmax(y[0], axis=-1), labels)
		output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

		try:
			output = self.bridge.cv2_to_imgmsg(output, "bgr8")
			self.image_pub.publish(output)
		except CvBridgeError as e:
			print(e)






def main():
	# Workaround to forbid tensorflow from crashing the gpu
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	K.set_session(sess)
	
	rospy.init_node("seg_inference_engine", anonymous=True)
	node = seg_node()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main()
	
	




