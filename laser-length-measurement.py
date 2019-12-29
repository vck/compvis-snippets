# https://github.com/alkasm/cvtools/blob/master/cvtools/hough.py
import cv2 as cv
import sys 
import numpy as np 
import imutils
from imutils import contours, perspective
import matplotlib.pyplot as plt 


def select_contour(counts):
	return counts[0], counts[len(counts)-1]

def find_x_y(filename):

	src = cv.imread(filename)
	depth = np.copy(src)

	gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
	gray = cv.bitwise_not(gray)
	bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -3)

	horizontal = np.copy(bw)
	vertical = np.copy(bw)
	cols = horizontal.shape[1]
	horizontal_size = int(cols/20)
	
	horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
	
	# Apply morphology operations
	horizontal = cv.erode(horizontal, horizontalStructure)
	horizontal = cv.dilate(horizontal, horizontalStructure)
	
	rows = vertical.shape[0]
	verticalsize = int(rows / 20)
	verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
	# Apply morphology operations
	vertical = cv.erode(vertical, verticalStructure)
	vertical = cv.dilate(vertical, verticalStructure)
	# vertical contour count
	target = vertical.copy()
	cnts = cv.findContours(target, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
	cnts = select_contour(cnts)

	points = []

	if len(cnts) == 2:
		rect = cv.minAreaRect(cnts[0])
		box = cv.boxPoints(rect)
		x, y, w, h = box
		points.append(x) 
		rect = cv.minAreaRect(cnts[1])
		box = cv.boxPoints(rect)
		x, y, w, h = box
		points.append(h)	


	# edge coordinate
	p1 = points[0]
	p2 = points[1]

	p1_coord = (int(p1[0]), int(p1[1])) # line coordinate
	p2_coord = (int(p2[0]), int(p2[1])) # line coordinate

	cv.line(src, (p1_coord), (p2_coord), (0, 100, 100), 2)

	D = p2[1] - p1[1]
	cal_d = D/35*2.1

	print("Jarak Patahan Vertikal")
	print("Jarak (pixel)", "\t", D)
	print("Jarak (cm)", "\t", cal_d)
	print()

	for c in cnts:
		rect = cv.minAreaRect(c)
		box = cv.boxPoints(rect)
		x, y, w, h = box
		# tampilkan garis laser hasil segmentasi
		segment = cv.boundingRect(c)
		x,y,w,h = segment
		
		cv.rectangle(src, (x, y), (x+w, y+h), (0, 255, 0), 2)

		for p in box:
			pt = (p[0], p[1])
			cv.circle(src,pt,1,(0, 200, 0),2)

	target = horizontal.copy()

	cnts = cv.findContours(target, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	(cnts, _) = contours.sort_contours(cnts)
	cnts = select_contour(cnts)
	a_area, b_area = cv.contourArea(cnts[0]), cv.contourArea(cnts[1])

	points = []

	if len(cnts) == 2:
		# first contour
		rect = cv.minAreaRect(cnts[0])
		box = cv.boxPoints(rect)
		x, y, w, h = box
		points.append(x) 

		# second contour
		rect = cv.minAreaRect(cnts[1])
		box = cv.boxPoints(rect)
		x, y, w, h = box
		points.append(y)	

	for c in cnts:
		rect = cv.minAreaRect(c)
		box = cv.boxPoints(rect)
		x, y, w, h = box
		points.append(x) 

		rect = cv.minAreaRect(cnts[1])
		box = cv.boxPoints(rect)
		
		x, y, w, h = box
		points.append(y)

	for c in cnts:		
		rect = cv.minAreaRect(c)
		box = cv.boxPoints(rect)
		box = perspective.order_points(box)
		segment = cv.boundingRect(c)
		x, y, w, h = segment
		cv.rectangle(src, (x, y),(x+w, y+h), (0, 255, 0), 2)
		
		for p in box:
			pt = (p[0], p[1])
			cv.circle(src, pt ,1,(250, 0, 0),2)

	p1 = points[0]
	p2 = points[1]

	p1_coord = (int(p1[0]), int(p1[1]))
	p2_coord = (int(p2[0]), int(p2[1]))

	cv.line(src, (p1_coord), (p2_coord), (100, 0, 100), 2)

	D = p2[0] - p1[0]
	cal_d = (D/35) * 2.1

	print("Patahan Horizontal")
	print("Jarak (pixel)", "\t", D)
	print("Jarak (cm)", "\t", cal_d)
	print()
	
	filename = filename.split(".")
	filename = filename[0]+"-points.jpg"
	
	cv.imwrite(filename, src)
	plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
	plt.show()

def main():
	filename = sys.argv[1]
	find_x_y(filename)

if __name__ == '__main__':
	main()
