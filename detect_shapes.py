# import the necessary packages
import imutils
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

# WIP: This isn't as straightforward as it seemed to be
def findThresh(gray):

	# Experimentar adaptive threshold, ou aumentar contraste em vez de gaussian
	
	hi_contrast = np.zeros(gray.shape, gray.dtype)
	for line in range(gray.shape[0]):
		for col in range(gray.shape[1]):
			hi_contrast[line, col] = np.clip(1.0 * gray[line, col], 0, 255)
	thresh, otsu = cv2.threshold(hi_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	print(thresh)
	cv2.imshow("Otsu", otsu)
	cv2.waitKey(0)
	return int(thresh)
	# return otsu

	# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 0)
	# cv2.imshow("Adaptive", binary)

def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 


def shape_detection(inverted_image):
	
	# Convert to HSV colourspace
	hsv = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2HSV)

	# Convert to grayscale	
	gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

	thresh = findThresh(gray)
	
	# Define the limits of the "black" colour <------ TODO: Definir Value de acordo com a imagem em vez de hard-coded (if possible)
	black_lo = np.array([0, 0, 0])
	black_hi = np.array([360, 255, thresh])

	# Create mask to select "blacks"
	mask = cv2.inRange(hsv, black_lo, black_hi)

	# Change "blacks" to pure black and "whites" to pure white
	inverted_image[mask > 0] = (0, 0, 0)
	inverted_image[mask <= 0] = (255, 255, 255)

	# Resize de image to a decent size
	resized = imutils.resize(inverted_image, width=1000)
	ratio = inverted_image.shape[0] / float(resized.shape[0])
	
	# convert the resized inverted_image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
	
	# find contours in the thresholded inverted_image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	allCenterx = []
	allCentery = []

	#Set Pixel Colors according Black and White regions
	imageWidth = inverted_image.shape[1] #Get image width
	imageHeight = inverted_image.shape[0] #Get image height

	yPos = int(imageHeight/2)
	xPos = 0

	barCount = 0
	foundFirstBlack = False
	offSet = 0

	pixelsColorLine = []

	while xPos < imageWidth and barCount < 30: #Loop through collumns
		if set(inverted_image[yPos, xPos]) == set([255,255,255]):
			if not foundFirstBlack:
				foundFirstBlack = True
				offSet = xPos
			pixelsColorLine.append([0, 0, 255])
		elif foundFirstBlack:
			pixelsColorLine.append([255, 153, 51])
			if set(inverted_image[yPos, xPos - 1]) == set([255,255,255]):
				barCount = barCount + 1

		xPos = xPos + 1 #Increment X position by 1

	xPos = 0

	# loop over the contours
	for c in cnts:
	
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours on the inverted_image
		c = c.astype("float")
		c *= ratio
		c = c.astype("int")

		# Detect the smallest bounding rectangle of the contour,
		# allowing the rectangle to be rotated.
		# (This could prove useful when bars are askew)
		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		# Detect the bounding rectangle of the contour
		# x and y are the coordinates of the top left vertice
		# w and h are the width and height, respectively
		(x, y, w, h) = cv2.boundingRect(c)
		# print((x, y, w, h))

		# Draw countours
		# print(w,h)
		# print(box)

		dist1 = math.sqrt((box[0,0] - box[1,0])**2 + (box[0,1] - box[1,1])**2)
		dist2 = math.sqrt((box[1,0] - box[2,0])**2 + (box[1,1] - box[2,1])**2)

		if (dist1 == 0 or dist2 == 0):
			continue

		print(dist1/dist2, dist2/dist1)
		
		if (dist1/dist2 >= 8) or (dist2/dist1 >= 8):
			allCenterx.append(x + int(w/2))
			allCentery.append(y + int(h/2))
			# Paint the center of the bounding rectangle
			cv2.circle(inverted_image, (x + int(w/2), y + int(h/2)), 1, (255, 0, 0), 2)
			# Draw the smallest bounding rectangle for now
			cv2.drawContours(inverted_image, [box], -1, (0, 255, 0), 1)
	
	# show the output inverted_image
	cv2.imshow("Image Inverted (Black & White)", inverted_image)
	cv2.waitKey(0)

	return pixelsColorLine, offSet