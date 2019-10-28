# USAGE
# python detect_barcode.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import detect_shapes

# Parses argument and returns image and image in grayscale
def loadImage():	
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
		help = "path to the image file")
	args = vars(ap.parse_args())

	# load the image and convert it to grayscale
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image, gray


def MainAlgorithm(gray):
	# compute the Scharr gradient magnitude representation of the images
	# in both the x and y direction using OpenCV 2.4
	ddepth = cv2.CV_32F
	gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
	gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

	# subtract the y-gradient from the x-gradient
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)

	# blur and threshold the image
	blurred = cv2.blur(gradient, (9,9)) 	
	(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

	# construct a closing kernel and apply it to the thresholded image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 7)) 
	aux = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	concat1 = np.hstack((blurred, aux))

	# perform a series of erosions and dilations
	closed = cv2.erode(aux, None, iterations = 4)
	closed = cv2.dilate(closed, None, iterations = 4)

	concat2 = np.hstack((concat1, closed))
	return closed, concat2

# shows concatenated images in a window
def showProgressInWindow(image):
	screen_res = 1440, 1080
	scale_width = screen_res[0] / image.shape[1]
	scale_height = screen_res[1] / image.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(image.shape[1] * scale)
	window_height = int(image.shape[0] * scale)
	cv2.namedWindow('Progress', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Progress', window_width, window_height)
	cv2.imshow('Progress', image)
	cv2.waitKey(0)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
def findCountors(closed):
	cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
	box = np.int0(box)
	return box

# Crops ROI of image
def crop(img, box):
	x1 = box[1][0]
	y1 = box[1][1]
	x2 = box[3][0]
	y2 = box[3][1]
		
	cropped = img[y1:y2, x1:x2]
	cv2.imshow("cropped", cropped)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return cropped


def Main():
	(image, gray) = loadImage()

	(closed, concatImages) = MainAlgorithm(gray)
	showProgressInWindow(concatImages)
	box = findCountors(closed)
	cropped = crop(image, box)

	inverted_image = cv2.bitwise_not(cropped)
	cv2.imshow("inverted", inverted_image)
	cv2.waitKey(0)
	

	detect_shapes.shape_detection(inverted_image)


	# draw a bounding box arounded the detected barcode and display the image
	cv2.drawContours(image, [box], -1, (0, 255, 0), 1)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	cv2.destroyWindow("Image")	

Main()