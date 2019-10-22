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

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])

# Converter para o colourspace HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Definir os limites da cor "Preto" <------ TODO: Definir valores de acordo com a imagem em vez de hard-coded (if possible)
black_lo = np.array([0, 0, 0])
black_hi = np.array([360, 150, 150])
# Criar mask ara selecionar "pretos"
mask = cv2.inRange(hsv, black_lo, black_hi)
# Mudar os "pretos" para preto puro e "brancos" para branco puro
image[mask > 0] = (0, 0, 0)
image[mask <= 0] = (255, 255, 255)

# Este resize funciona melhor com valores altos
resized = imutils.resize(image, width=1000)
ratio = image.shape[0] / float(resized.shape[0])
 
# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
 
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
 
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 1)

	# Detect the bounding rectangle of the contour
	# x and y are the coordinates of the top left vertice
	# w and h are the width and height, respectively
	(x, y, w, h) = cv2.boundingRect(c)
	print((x, y, w, h))
 
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
