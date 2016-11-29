import os
import numpy as np
import cv2
from networktables import NetworkTable

os.system("sudo bash /home/pi/vision/init.sh")

NetworkTable.setIPAddress("roboRIO-4914-FRC.local")
NetworkTable.setClientMode()
NetworkTable.initialize()
table = NetworkTable.getTable("ContoursReport")

COLOR_MIN = np.array([60, 100, 100])
COLOR_MAX = np.array([85, 255, 255])
MIN_AREA = 250
CAM_ID = 0
DEBUG = False

cap = cv2.VideoCapture(CAM_ID)

while True:
	# read image from camera
	ret, frame = cap.read()

	# resize image to 320x240
	frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
	
	# convert BGR format to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if DEBUG:
		cv2.imshow('hsv', hsv)
		cv2.imshow('brg', frame)

	# threshold HSV image based on HSV ranges given by COLOR_MIN and COLOR_MAX
	frame = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)

	if DEBUG:
		cv2.imshow('frame', frame)

	# find contours based on thresholded image
	_, contours, heirarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# clear array of contours from previous iteration
	filteredContours = []

	# removes contours smaller than minimum area
	for i in range(0, len(contours)):
		if cv2.contourArea(contours[i]) > MIN_AREA:
			filteredContours.append(contours[i])

	# processes largest filtered contour by area if present
	if len(filteredContours) > 0:
		# default largest contour index and max area
		iLargestContour = 0;
		maxArea = 0;

		# searches for index of largest contour by area
		for i in range(0, len(filteredContours):
			if cv2.contourArea(filteredContours[i]) > maxArea:
				maxArea = cv2.contourArea(filteredContours[i])
				iLargestContour = i

		# largest contour
		c = filteredContours[iLargestContour]

		# calculates centers in the X and Y axes of image
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		# prints center X and center Y to console for debug purposes
		print("cX:", repr(cX).rjust(3), " cY:", repr(cY).rjust(3))

		# publishes contour values to networkTable ContoursReport
		table.putNumber('isTarget', 1)
		table.putNumber('cX', cX)
		table.putNumber('cY', cY)

	# publishes default values to table if no target found
	else:
		# publishes default no target values to networkTable ContoursReport
		table.putNumber('isTarget', 0)
		table.putNumber('cX', -1)
		table.putNumber('cY', -1)

	if DEBUG:
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if DEBUG:
	cap.release()
	cv2.destroyAllWindows()
