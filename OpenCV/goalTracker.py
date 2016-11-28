import sys
# sys.path.append('/home/pi/.virtualenvs/cv/lib/python3.4/site-packages')
import os
os.system("sudo bash /home/pi/vision/init.sh")

import numpy as np
import cv2

COLOR_MIN = np.array([60, 100, 100])
COLOR_MAX = np.array([85, 255, 255])
MIN_AREA = 175
CAM_ID = 0
DEBUG = False

cap = cv2.VideoCapture(CAM_ID)
#cap.release()
#cap.open(CAM_ID)

while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if DEBUG:
		cv2.imshow('hsv', hsv)
		cv2.imshow('brg', frame)

	frame = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)

	if DEBUG:
		cv2.imshow('frame', frame)

	_, contours, heirarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	filteredContours = []

	for i in range(0, len(contours)):
		if cv2.contourArea(contours[i]) > MIN_AREA:
			filteredContours.append(contours[i])

	if len(filteredContours) > 0:
		c = filteredContours[0]

		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		print(cX)
		print(cY)
		print("----")
	if DEBUG:
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if DEBUG:
	cap.release()
	cv2.destroyAllWindows()
