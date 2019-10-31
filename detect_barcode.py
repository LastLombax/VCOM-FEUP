# USAGE
# python detect_barcode.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import math
import detect_shapes
from PIL import Image, ImageStat

def detect_color_image(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
	thresh = 225
	pil_img = Image.open(file)
	bands = pil_img.getbands()
	if bands == ('R','G','B') or bands== ('R','G','B','A'):
		thumb = pil_img.resize((thumb_size,thumb_size))
		SSE, bias = 0, [0,0,0]
		if adjust_color_bias:
			bias = ImageStat.Stat(thumb).mean[:3]
			bias = [b - sum(bias)/3 for b in bias ]
		for pixel in thumb.getdata():
			mu = sum(pixel)/3
			SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
		MSE = float(SSE)/(thumb_size*thumb_size)
		if MSE <= MSE_cutoff:
			thresh = 50
	elif len(bands) == 1:
		thresh = 50

	return thresh

			

# Parses argument and returns image and image in grayscale
def loadImage():	
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
		help = "path to the image file")
	args = vars(ap.parse_args())
	thresh = detect_color_image(args["image"])

	# load the image and convert it to grayscale
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image, gray, thresh


def MainAlgorithm(gray, calc_thresh):
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
	(_, thresh) = cv2.threshold(blurred, calc_thresh, 255, cv2.THRESH_BINARY)

	# construct a closing kernel and apply it to the thresholded image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 7)) 
	aux = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	concat1 = np.hstack((blurred, aux))

	# perform a series of erosions and dilations
	closed = cv2.erode(aux, None, iterations = 6)
	closed = cv2.dilate(closed, None, iterations = 6)

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
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
	
	print("Extents")
	for c in cnts:		
		area = cv2.contourArea(c)
		x,y,w,h = cv2.boundingRect(c)
		rect_area = w*h
		extent = float(area)/rect_area
		print(extent)
		if extent > 0.6:
			break

	# VER 9, 16, 18(mais ou menos), 19, 20, 21, 25
	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
	box = np.int0(box)
	x, y, w, h = cv2.boundingRect(box)
	return box, x, y, w, h

# Returns the rotated rectangle's angle in relation to the X axis
def getRotationAngle(box):
	dist1 = math.sqrt((box[0,0] - box[1,0])**2 + (box[0,1] - box[1,1])**2)
	dist2 = math.sqrt((box[1,0] - box[2,0])**2 + (box[1,1] - box[2,1])**2)

	angle = 0
	if (dist1 > dist2):
		angle = -math.acos( (box[0,0] - box[1,0]) / dist1)
	else:
		angle = math.asin( (box[0,0] - box[1,0]) / dist1)
	return angle * 180 / math.pi

# Crops ROI of image
def crop(img, box, rectx, recty, w, h):
	
	cropped = np.zeros((h, w, img.shape[2]), img.dtype)
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if (cv2.pointPolygonTest(box, (x,y), False) >= 0):
				cropped[y - recty, x - rectx] = img[y,x]

	cv2.imshow("Cropped image", cropped)
	cv2.waitKey(0)

	angle = getRotationAngle(box)
	cropped = imutils.rotate_bound(cropped,angle)

	cv2.imshow("Rotated image", cropped)
	cv2.waitKey(0)

	cv2.imshow("cropped", cropped)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return cropped


def Main():
	(image, gray, thresh) = loadImage()

	(closed, concatImages) = MainAlgorithm(gray, thresh)
	showProgressInWindow(concatImages)
	box, rectx, recty, w, h  = findCountors(closed)

	cropped = crop(image, box, rectx, recty, w, h)

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