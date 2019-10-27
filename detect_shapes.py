# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# Convert to HSV colourspace
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the limits of the "black" colour <------ TODO: Definir Value de acordo com a imagem em vez de hard-coded (if possible)
black_lo = np.array([0, 0, 0])
black_hi = np.array([360, 255, 150])

# Create mask to select "blacks"
mask = cv2.inRange(hsv, black_lo, black_hi)

# Change "blacks" to pure black and "whites" to pure white
image[mask > 0] = (0, 0, 0)
image[mask <= 0] = (255, 255, 255)

# Resize de image to a decent size
resized = imutils.resize(image, width=1000)
ratio = image.shape[0] / float(resized.shape[0])
 
# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
 
# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
 
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")

	# Detect the smallest bounding rectangle of the contour,
	# allowing the rectangle to be rotated.
	# (This could prove useful when bars are askew)
	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	box = np.int0(box)

	# Draw the smallest bounding rectangle for now
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

	# Detect the bounding rectangle of the contour
	# x and y are the coordinates of the top left vertice
	# w and h are the width and height, respectively
	(x, y, w, h) = cv2.boundingRect(c)
	print((x, y, w, h))

	# Paint the center of the bounding rectangle
	cv2.circle(image, (x + int(w/2), y + int(h/2)), 1, (255, 0, 0), 2)
 
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
