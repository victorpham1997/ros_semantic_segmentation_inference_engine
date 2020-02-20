# Semantic Segmentation engine for ROS

### Dependencies:

1. Python 2.7 
2. CUDA 10.0
3. cuDNN 7
4. tensorflow-gpu 1.14.0 (for python2.7)

### Important:

- Change the weights and config json respectively if updated

### Usage:

- Copy the package into your catkin_ws folder
- run catkin_make
- rosrun ros_semantic_segmen_inference_engine inference_node.py 
- Segmented image is published in /seg_img topic

