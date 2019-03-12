# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import pyrealsense2 as rs

import numpy as np


# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append('/home/cadmin/Documents/openpose/python');
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="/home/cadmin/Documents/openpose/examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/home/cadmin/Documents/openpose/models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Config
#proto_file = os.path.abspath("./resources/pose_deploy_linevec.prototxt")
#weights_file = os.path.abspath("./resources/pose_iter_440000.caffemodel")

#net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

CAM_WIDTH = 640
CAM_HEIGHT = 480
image_height = 224
image_width = 224 # int((image_height / frame_height) * frame_width)
frame_width = 224
frame_height = 224
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

# Read Webcam
pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    in_blob = cv2.dnn.blobFromImage(color_image, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)
    # net.setInput(in_blob)
    # output = net.forward()

    datum = op.Datum()
    datum.cvInputData = color_image
    opWrapper.emplaceAndPop([datum])

    # H = output.shape[2]
    # W = output.shape[3]

    cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break










# Read frames on directory
# imagePaths = op.get_images_on_directory(args[0].image_dir);
# start = time.time()

# # Process and display images
# for imagePath in imagePaths:
#     datum = op.Datum()
#     imageToProcess = cv2.imread(imagePath)
#     datum.cvInputData = imageToProcess
#     opWrapper.emplaceAndPop([datum])

#     print("Body keypoints: \n" + str(datum.poseKeypoints))

#     if not args[0].no_display:
#         cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
#         key = cv2.waitKey(15)
#         if key == 27: break

# end = time.time()
# print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
