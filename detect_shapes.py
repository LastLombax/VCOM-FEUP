# import the necessary packages
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

# WIP: This isn't as straightforward as it seemed to be
def findMidValue(hsvImage):
	
	rows, cols, chan = hsvImage.shape
	values = []
	
	for i in range(rows):
		for j in range(cols):
			values.append(hsvImage[i,j,2])
	values.sort()

	# print(hsvImage)

	maxDiff = 0
	maxDiffIdx = 0
	for i in range(len(values) - 1):
		if values[i + 1] - values[i] > maxDiff:
			maxDiff = values[i + 1] - values[i]
			maxDiffIdx = i

	return values[maxDiffIdx] + int(maxDiff / 2)

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


def shape_detection(image):
	
	# Convert to HSV colourspace
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	# Define the limits of the "black" colour <------ TODO: Definir Value de acordo com a imagem em vez de hard-coded (if possible)
	black_lo = np.array([0, 0, 0])
	black_hi = np.array([360, 255, 128])

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
	allCenterx = []
	allCentery = []

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

		# Detect the bounding rectangle of the contour
		# x and y are the coordinates of the top left vertice
		# w and h are the width and height, respectively
		(x, y, w, h) = cv2.boundingRect(c)
		# print((x, y, w, h))

		# Paint the center of the bounding rectangle
		print(w,h)
		if h/w >= 10 and h/w <= 50:
			print("detetou...")
			allCenterx.append(x + int(w/2))
			allCentery.append(y + int(h/2))
			cv2.circle(image, (x + int(w/2), y + int(h/2)), 1, (255, 0, 0), 2)
			# Draw the smallest bounding rectangle for now
			cv2.drawContours(image, [box], -1, (0, 255, 0), 1)

	
		# show the output image
		cv2.imshow("Image", image)
		cv2.waitKey(0)

	# estimating coefficients 
	b = estimate_coef(np.asarray(allCenterx), np.asarray(allCentery))
	b2=estimate_coef(np.asarray(allCenterx), np.asarray(allCentery))

	# calculate axes intersection points
	a1x = np.float32(-b[0]/b[1])
	a1y = 0
	a2x = 0
	a2y = np.float32(b[0])

	 # predicted response vector 
    # y_pred = b[0] + b[1]*x

	y= b[0] + b[1]*146

	# plotting regression line 
	lineThickness = 2
	print(a1x, a1y, a2x, a2y)
	cv2.line(image, (a2x, a2y), (146,int(y)), (200,0,0), lineThickness)

	# show the output image
	cv2.imshow("ImageLinearRegression", image)
	cv2.waitKey(0)
	plot_regression_line(np.asarray(allCenterx), np.asarray(allCentery), b2) 