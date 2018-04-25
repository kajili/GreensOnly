from skimage.measure import compare_ssim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import imutils
import cv2
import os

image_list = []

original1 = cv2.imread("images/green8.jpg")
#print (original.shape)
#compare = cv2.imread("images/green3.jpg")

original = cv2.cvtColor(original1, cv2.COLOR_BGR2GRAY)
#compmare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

#display image with the one that most similar to the original in gray
def compare_images(imageA, imageB, title, s):

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("SSIM: %.2f" % s)
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()

# Work on progress
# Run thru a folder
percent = 0

#The path for my folder
path = "C:/Users/silve/Desktop/atollogy/GreensOnly/images/23/"
for filename in os.listdir(path):
	#only targeting the jpg images
	if filename.endswith(".jpg"):
		#Combine the jpg filename with path
		filename = os.path.join(path, filename)
		#printing the filename to check
		#print(filename)
		#get the image
		compare = cv2.imread(filename)
		#print (compare.shape)
		#Turn image gray
		gray_compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)
		#Get the decimal rate
		(s, d) = compare_ssim(original, gray_compare, full=True)
		if s> percent:
			percent = s
			similar = gray_compare
			print(filename)
			print(percent)
			compare_images(original1, similar, "Original vs Compare", percent)