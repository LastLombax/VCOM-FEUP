# USAGE
# python detect_barcode.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

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
	(_, thresh) = cv2.threshold(blurred, 223, 255, cv2.THRESH_BINARY)

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
	cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Resized Window', window_width, window_height)
	cv2.imshow('Resized Window', image)

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

def stretch(img, box):
	# reshape borders
	pts = box.reshape(4,2)
	rect = np.zeros((4, 2), dtype="float32")
	

	# the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	
	# compute the difference between the points -- the top-right
    # will have the minimum difference and the bottom-left will
    # have the maximum difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

    # multiply the rectangle by the original ratio
	rect *= 1

	 # now that we have our rectangle of points, let's compute
    # the width of our new image
	(tl, tr, br, bl) = rect
	print(rect)
	width_a = width_b = np.sqrt(((br[0] - bl[0]) * 2) + ((br[1] - bl[1]) * 2))

	# ...and now for the height of our new image
	height_a = height_b = np.sqrt(((tr[0] - br[0]) * 2) + ((tr[1] - br[1]) * 2))

	# take the maximum of the width and height values to reach
    # our final dimensions

	max_width = max(int(width_a), int(width_b))
	max_height = max(int(height_a), int(height_b))

	# construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
	dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

	 # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
	M = cv2.getPerspectiveTransform(rect, dst)
	return cv2.warpPerspective(img, M, (max_width, max_height)) 


def Main():
	(image, gray) = loadImage()
	(closed, concatImages) = MainAlgorithm(gray)
	showProgressInWindow(concatImages)
	box = findCountors(closed)

	# draw a bounding box arounded the detected barcode and display the image
	cv2.drawContours(image, [box], -1, (0, 255, 0), 1)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

	warp = stretch(image, box)
	cv2.imshow("Warp", warp)
	cv2.waitKey(0)

	
	



Main()