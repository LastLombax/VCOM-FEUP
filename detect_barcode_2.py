import numpy as np
import cv2
import imutils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# equalize lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# edge enhancement
edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, 
                         ksize = 3, scale = 1, delta = 0)


# bilateral blur, which keeps edges
blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)

# use simple thresholding. adaptive thresholding might be more robust
(_, thresh) = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
#thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
 #           cv2.THRESH_BINARY,5,2)



concat1 = np.hstack((edge_enh, thresh))


# do some morphology to isolate just the barcode blob
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)



concat2 = np.hstack((concat1, closed))

# Shows all in one window
screen_res = 1440, 1080
scale_width = screen_res[0] / concat2.shape[1]
scale_height = screen_res[1] / concat2.shape[0]
scale = min(scale_width, scale_height)
window_width = int(concat2.shape[1] * scale)
window_height = int(concat2.shape[0] * scale)
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Window', window_width, window_height)

cv2.imshow('Resized Window', concat2)


# find contours left in the image
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
print(box)
cv2.imshow("found barcode", image)
cv2.waitKey(0)
