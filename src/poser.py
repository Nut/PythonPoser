import cv2
import numpy as np

frame_width = 800
frame_height = 600

proto_file = "../resources/pose_deploy_linevec.prototxt"
weights_file = "../pose_iter_440000.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

image1 = cv2.imread("../resources/sample.jpg")
dimensions = image1.shape
image_height = dimensions[0]
image_width = int((image_height / frame_height) * frame_width)

in_blob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

net.setInput(in_blob)
output = net.forward()

i = 0
prob_map = output[0, i, :, :]
prob_map = cv2.resize(prob_map, (frame_width, frame_height))


cv2.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
cv2.imshow(prob_map, alpha=0.6)