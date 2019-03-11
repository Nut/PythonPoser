import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

CAM_WIDTH = 224
CAM_HEIGHT = 224

proto_file = os.path.abspath("./resources/pose_deploy_linevec.prototxt")
weights_file = os.path.abspath("./resources/pose_iter_440000.caffemodel")

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


image1 = cv2.imread(os.path.abspath("./resources/sample.jpg"))
dimensions = image1.shape
image_height = CAM_HEIGHT
image_width = CAM_WIDTH # int((image_height / frame_height) * frame_width)
frame_width = CAM_WIDTH 
frame_height = CAM_HEIGHT


# plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# plt.imshow(prob_map, alpha=0.6)
plt.show()

while True:
	ret, frame = cam.read()

	in_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

	net.setInput(in_blob)
	output = net.forward()

	i = 0
	prob_map = output[0, i, :, :]
	prob_map = cv2.resize(prob_map, (frame_width, frame_height))

	cv2.imshow("Window", frame)
	cv2.imshow("Map", prob_map)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()