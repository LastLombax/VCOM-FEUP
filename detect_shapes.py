# import the necessary packages
import imutils
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

# Augments image's threshold to use in otsu's algorithm and returns calculated threshold
def findThresh(gray):

	hi_contrast = np.zeros(gray.shape, gray.dtype)
	for line in range(gray.shape[0]):
		for col in range(gray.shape[1]):
			hi_contrast[line, col] = np.clip(1.0 * gray[line, col], 0, 255)
	thresh, otsu = cv2.threshold(hi_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	return int(thresh)

# Detects the shapes of the bars and spaces
def shape_detection(inverted_image):
	
	# Convert to HSV colourspace
	hsv = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2HSV)

	# Convert to grayscale	
	gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

	thresh = findThresh(gray)
	
	# Define the limits of the "black" colour 
	black_lo = np.array([0, 0, 0])
	black_hi = np.array([360, 255, thresh])

	# Create mask to select "blacks"
	mask = cv2.inRange(hsv, black_lo, black_hi)

	# Change "blacks" to pure black and "whites" to pure white
	inverted_image[mask > 0] = (0, 0, 0)
	inverted_image[mask <= 0] = (255, 255, 255)

	resized = imutils.resize(inverted_image, width=1000)
	ratio = inverted_image.shape[0] / float(resized.shape[0])
	

	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
	
	# find contours in the thresholded inverted_image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	allCenterx = []
	allCentery = []

	# Set Pixel Colors according Black and White regions
	imageWidth = inverted_image.shape[1] 
	imageHeight = inverted_image.shape[0] 

	yPos = int(imageHeight/2)
	xPos = 0

	barCount = 0
	foundFirstBlack = False
	offSet = 0

	pixelsColorLine = []

	while xPos < imageWidth and barCount < 30: 
		if set(inverted_image[yPos, xPos]) == set([255,255,255]):
			if not foundFirstBlack:
				foundFirstBlack = True
				offSet = xPos
			pixelsColorLine.append([0, 0, 255])
		elif foundFirstBlack:
			pixelsColorLine.append([255, 153, 51])
			if set(inverted_image[yPos, xPos - 1]) == set([255,255,255]):
				barCount = barCount + 1

		xPos = xPos + 1 

	xPos = 0

	for c in cnts:

		c = c.astype("float")
		c *= ratio
		c = c.astype("int")

		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		(x, y, w, h) = cv2.boundingRect(c)

		dist1 = math.sqrt((box[0,0] - box[1,0])**2 + (box[0,1] - box[1,1])**2)
		dist2 = math.sqrt((box[1,0] - box[2,0])**2 + (box[1,1] - box[2,1])**2)

		if (dist1 == 0 or dist2 == 0):
			continue
		
		if (dist1/dist2 >= 8) or (dist2/dist1 >= 8):
			allCenterx.append(x + int(w/2))
			allCentery.append(y + int(h/2))
			
			cv2.circle(inverted_image, (x + int(w/2), y + int(h/2)), 1, (255, 0, 0), 2)
			
			cv2.drawContours(inverted_image, [box], -1, (0, 255, 0), 1)
	

	return pixelsColorLine, offSet