import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pyrealsense2 as rs

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

	pipeline = rs.pipeline()
	
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

	pipeline.start(config)
	#plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
	#plt.imshow(prob_map, alpha=0.6)
	#plt.show()

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

		i = 0
		prob_map = output[0, i, :, :]
		prob_map = cv2.resize(prob_map, (CAM_WIDTH, CAM_HEIGHT))
		map_image = np.asanyarray(prob_map)

		# Stack both images horizontally
		images = np.hstack((color_image,  depth_colormap))
		
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', images)
		cv2.namedWindow("Map", cv2.WINDOW_AUTOSIZE)
		cv2.imshow('Map', map_image)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
finally:
	pipeline.stop()
	cv2.destroyAllWindows()