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
params["model_pose"] = "BODY_25"

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

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Config
CAM_WIDTH = 640
CAM_HEIGHT = 480
image_height = 224
image_width = 224 # int((image_height / frame_height) * frame_width)
frame_width = 224
frame_height = 224
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

POSE_BODY_25_BODY_PARTS = {
    "Nose" : 0,
    "Neck" : 1,
    "RShoulder" : 2,
    "RElbow" : 3,
    "RWrist" : 4,
    "LShoulder" : 5,
    "LElbow" : 6,
    "LWrist" : 7,
    "MidHip" : 8,
    "RHip" : 9,
    "RKnee" : 10,
    "RAnkle" : 11,
    "LHip" : 12,
    "LKnee" : 13,
    "LAnkle" : 14,
    "REye" : 15,
    "LEye" : 16,
    "REar" : 17,
    "LEar" : 18,
    "LBigToe" : 19,
    "LSmallToe" : 20,
    "LHeel" : 21,
    "RBigToe" : 22,
    "RSmallToe" : 23,
    "RHeel" : 24,
    "Background" : 25
}

POSE_PAIRS_25 = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [2,17], [5,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

# Functions
def getKeyPointCoords(pose_keypoints, depth_frame, depth_colormap):
    keypoints_out = []
    pose_keypoints = pose_keypoints.tolist()
    if type(pose_keypoints) == float:
        return keypoints_out, depth_colormap

    for i in range(0, len(pose_keypoints)):
        keypoints_out.append([])
        person = pose_keypoints[i]
        for j in range(0, len(person)):
            point = person[j]
            x = float(point[0])
            y = float(point[1])
            z = depth_frame.get_distance(int(x), int(y))
            keypoints_out[i].append([x, y, z])
            cv2.circle(depth_colormap, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    return keypoints_out, depth_colormap

# Read Reference Recording from file


# Read Webcam
pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)

    datum = op.Datum()
    datum.cvInputData = color_image
    opWrapper.emplaceAndPop([datum])

    output = datum.poseKeypoints

    coords, depth_colormap = getKeyPointCoords(output, depth_frame, depth_colormap)
    
    images = np.hstack((datum.cvOutputData,  depth_colormap))
    cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
