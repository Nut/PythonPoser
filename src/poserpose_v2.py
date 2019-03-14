# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import pyrealsense2 as rs
import json
import math

import numpy as np
from numpy_ringbuffer import RingBuffer

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

INTRINSICS = None

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

def transformPixelToCameraCoords(coords): # Pixel to Camera Coordinates
    return rs.rs2_deproject_pixel_to_point(INTRINSICS, [coords[0], coords[1]], coords[2])

def getVector(coord_1, coord_2):
    a = np.array(coord_1)
    b = np.array(coord_2)
    return b - a

def getBodyBaseVectors(keypoints):
    base_vector_x = getVector(keypoints[8], keypoints[12])
    base_vector_y = getVector(keypoints[8], keypoints[1])

    if np.linalg.norm(base_vector_x) == 0:
        base_vector_x_norm = [0, 0, 0]
    else:
        base_vector_x_norm = base_vector_x / np.linalg.norm(base_vector_x)

    if np.linalg.norm(base_vector_y) == 0:
        base_vector_y_norm = [0, 0, 0]
    else:
        base_vector_y_norm = base_vector_y / np.linalg.norm(base_vector_y)

    base_vector_z_norm = np.cross(base_vector_x_norm, base_vector_y_norm)
    return base_vector_x_norm, base_vector_y_norm, base_vector_z_norm

def getAngle(vector_a, vector_b):
    return math.acos(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))

def transformCameraToBodyCoords(keypoints, base_keypoint, coord):
    rc = np.array(coord)
    hc = transformPixelToCameraCoords(base_keypoint)
    rotation_matrix = np.column_stack(getBodyBaseVectors(keypoints))
    vector = rotation_matrix.dot(rc - hc)
    return vector

def getKeyPointCoords(pose_keypoints, depth_frame, depth_colormap):
    keypoints_out = []
    pose_keypoints = pose_keypoints.tolist()
    if type(pose_keypoints) == float:
        return keypoints_out, depth_colormap

    for i in range(0, len(pose_keypoints)):
        keypoints_out.append([])
        person = pose_keypoints[i]
        for j in range(0, len(person)):
            if j not in [10, 11, 13, 14]: # only specific keypoints
                point = person[j]
                x = float(point[0])
                y = float(point[1])
                z = depth_frame.get_distance(int(x), int(y))
                cv2.circle(depth_colormap, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            else:
                x, y, z = float(0.0), float(0.0), float(0.0)

            keypoints_out[i].append([x, y, z])
    return keypoints_out, depth_colormap

def getReferenceData(file):
    imported_json = json.load(file)
    flattened_ref_keypoints = []
    for frame in imported_json:
        for person in frame:
            frame = []
            for keypoint in person:
                camera_coords = transformPixelToCameraCoords(keypoint)
                body_coords = transformCameraToBodyCoords(person, person[8], camera_coords)
                frame.append(body_coords)
            flattened_ref_keypoints.append(frame)
    return np.array(flattened_ref_keypoints)

# Read Reference Recording from file
with open("./resources/recording_pol.txt", "r") as file:
    # Read Webcam
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    ringbuffers = []
    # Add one ringbuffer for each keypoint
    for i in range(0, 24):
        ringbuffers.append(RingBuffer(capacity=28, dtype=list)) # capacity depends on length of recording

    prev_sum_distance = 0
    x = None

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        # calibriated internal parameters from RealSense
        INTRINSICS = depth_frame.profile.as_video_stream_profile().intrinsics
        if x is None:
            x = getReferenceData(file)

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

        coords_pixel, depth_colormap = getKeyPointCoords(output, depth_frame, depth_colormap)
        coords = [[]]

        image_frame = datum.cvOutputData

        if coords_pixel:
            for keypoint in coords_pixel[0]:
                coords[0].append(transformCameraToBodyCoords(coords_pixel[0], coords_pixel[0][8], transformPixelToCameraCoords(keypoint)))

            sum_distance = 0

            if coords:
                for buffer_num in range(0, 24):
                    if buffer_num not in [10, 11, 13, 14]: # only specific keypoints
                        x_temp, y_temp, z_temp = coords[0][buffer_num]
                        ringbuffers[buffer_num].append([x_temp, y_temp, z_temp])
                    else:
                        ringbuffers[buffer_num].append([0.0, 0.0, 0.0])
                    y = np.array(ringbuffers[buffer_num]).tolist()
                    distance, path = fastdtw(x[buffer_num], y, dist=euclidean)
                    sum_distance += distance
                print(sum_distance)

            distance_derivative = sum_distance - prev_sum_distance
        
            if distance_derivative < -10.0:
                print("KNIEBEUGE")
                cv2.rectangle(image_frame, (0, 0), (640, 480), (0, 255, 0), thickness=5)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_frame,'Kniebeuge',(10,400), font, 4,(0,255,0),2,cv2.LINE_AA)
        
        images = np.hstack((image_frame,  depth_colormap))
        cv2.imshow("PoserPose", images)

        prev_sum_distance = sum_distance

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
