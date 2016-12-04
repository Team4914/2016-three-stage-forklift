import os
import numpy as np
import cv2
import math
# from networktables import NetworkTable

os.system("sudo bash init.sh")

def cart2pol(a):
    x = a[0]
    y = a[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return([rho, phi])

def pol2cart(a):
    rho = a[0]
    phi = a[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return([x, y])

# NetworkTable.setIPAddress("roboRIO-4914-FRC.local")
# NetworkTable.setClientMode()
# NetworkTable.initialize()
# table = NetworkTable.getTable("ContoursReport")

COLOR_MIN = np.array([60, 100, 100])
COLOR_MAX = np.array([85, 255, 255])
MIN_AREA = 250
VIEW_ANGLE = 60 # (for lifecam 3000)
VIEW_ANGLE *= 360
VIEW_ANGLE /= 2*3.1415926535
TARGET_LENGTH = 51 # width of retroreflective tape, in cm
FOV_PIXEL = 320
CAM_ID = 1
DEBUG = True

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
	contours, heirarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
		for i in range(0, len(filteredContours)):
			if cv2.contourArea(filteredContours[i]) > maxArea:
				maxArea = cv2.contourArea(filteredContours[i])
				iLargestContour = i

		# largest contour
		c = filteredContours[iLargestContour]

		# calculates centers in the X and Y axes of image
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		# x,y,w,h = cv2.boundingRect(c)
    		# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		rect = cv2.minAreaRect(c)
		# box = cv2.boxPoints(rect)
		print (rect[1][0])

		wX = rect[1][0]/2
		wY = rect[1][1]/2

		xLeft = -wX
		xRight = wX
		yBottom = wY
		yTop = -wY

		topLeft = [xLeft, yTop]
		topRight = [xRight, yTop]
		bottomLeft = [xLeft, yBottom]
		bottomRight = [xRight, yBottom]

		theta = rect[2]/360
		theta *= 3.1415926535*2

		topLeft = cart2pol(topLeft)
		topRight = cart2pol(topRight)
		bottomLeft = cart2pol(bottomLeft)
		bottomRight = cart2pol(bottomRight)

		topLeft[1] += theta
		topRight[1] += theta
		bottomLeft[1] += theta
		bottomRight[1] += theta

		topLeft = pol2cart(topLeft)
		topRight = pol2cart(topRight)
		bottomLeft = pol2cart(bottomLeft)
		bottomRight = pol2cart(bottomRight)

		topLeft[0] += cX
		topLeft[1] += cY
		topRight[0] += cX
		topRight[1] += cY
		bottomLeft[0] += cX
		bottomLeft[1] += cY
		bottomRight[0] += cX
		bottomRight[1] += cY

		targetPixelLength = cv2.norm(bottomLeft, bottomRight)

		targetDistance = TARGET_LENGTH*FOV_PIXEL
		targetDistance /= 2*targetPixelLength*Math.tan(VIEW_ANGLE)

		print("TargetDistance:", targetDistance)

		# box = [topLeft, topRight, bottomRight, bottomLeft]		
		# box = np.int0(box)
		# cv2.drawContours(hsv, [box], 0, (0,0,255),2)

		# cv2.imshow('asdf', hsv)

		# prints center X and center Y to console for debug purposes
		print("cX:", repr(cX).rjust(3), " cY:", repr(cY).rjust(3))

		# publishes contour values to networkTable ContoursReport
		# table.putNumber('isTarget', 1)
		# table.putNumber('cX', cX)
		# table.putNumber('cY', cY)

	# publishes default values to table if no target found
	# else:
		# publishes default no target values to networkTable ContoursReport
		# table.putNumber('isTarget', 0)
		# table.putNumber('cX', -1)
		# table.putNumber('cY', -1)

	if DEBUG:
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if DEBUG:
	cap.release()
	cv2.destroyAllWindows()
