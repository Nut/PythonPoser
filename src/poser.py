import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pyrealsense2 as rs
import time

try:
	image1 = cv2.imread(os.path.abspath("./resources/sample.jpg"))
	dimensions = image1.shape

	proto_file = os.path.abspath("./resources/pose_deploy_linevec.prototxt")
	weights_file = os.path.abspath("./resources/pose_iter_440000.caffemodel")

	net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
	
	CAM_WIDTH = 640
	CAM_HEIGHT = 480
	image_height = 224
	image_width = 224 # int((image_height / frame_height) * frame_width)
	frame_width = 224
	frame_height = 224
	POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

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
		net.setInput(in_blob)
		output = net.forward()

		H = output.shape[2]
		W = output.shape[3]

	# Empty list to store the detected keypoints
		points = []
		for i in range(18):
			# confidence map of corresponding body's part.
			probMap = output[0, i, :, :]
		
			# Find global maxima of the probMap.
			minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
			
			# Scale the point to fit on the original image
			x = (CAM_WIDTH * point[0]) / W
			y = (CAM_HEIGHT * point[1]) / H
		
			if prob > 0.1 : 
				cv2.circle(color_image, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(color_image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
		
				# Add the point to the list if the probability is greater than the threshold
				points.append((int(x), int(y)))
			else :
				points.append(None)

		for pair in POSE_PAIRS:
			partA = pair[0]
			partB = pair[1]

			if points[partA] and points[partB]:
				cv2.line(color_image, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
				cv2.circle(color_image, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.circle(color_image, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

		'''i = 0
		prob_map = output[0, i, :, :]
		prob_map = cv2.resize(prob_map, (CAM_WIDTH, CAM_HEIGHT))
		map_image = np.asanyarray(prob_map)'''

		# Stack both images horizontally
		images = np.hstack((color_image,  depth_colormap))
		
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', images)



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
finally:
	pipeline.stop()
	cv2.destroyAllWindows()