import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


proto_file = os.path.abspath("./resources/pose_deploy_linevec.prototxt")
weights_file = os.path.abspath("./resources/pose_iter_440000.caffemodel")

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

image1 = cv2.imread(os.path.abspath("./resources/sample.jpg"))
dimensions = image1.shape
image_height = dimensions[0]
image_width = dimensions[1] # int((image_height / frame_height) * frame_width)
frame_width = dimensions[1]
frame_height = dimensions[0]

in_blob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

net.setInput(in_blob)
output = net.forward()

i = 0
prob_map = output[0, i, :, :]
prob_map = cv2.resize(prob_map, (frame_width, frame_height))

plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.imshow(prob_map, alpha=0.6)
plt.show()