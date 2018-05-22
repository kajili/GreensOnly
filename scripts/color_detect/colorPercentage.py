"""
" File:     colorPercentage.py
" Author:   An Tran  <anngtran@ucsc.edu>
" Date:     05-21-2018
"""
from PIL import Image
import argparse
import cv2

#usage python printColor.py -i example.jpg
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

image = Image.open(args["image"])

#getcolors() get a list of rgb in an image. return a list of (count, pixel)
colors = image.convert('RGB').getcolors(100000)

sortedColors = sorted(colors)

print(sortedColors)

#Total count of pixels in an image
count = 0;
for x in sortedColors:
	count = count + x[0] 
print(count)

#Percent from number of time a certain pixels appear over all the pixels in an image
for x in sortedColors:
	if(x[1] != (0, 0, 0)):
		print((x[0]/count)*100, "% of the pixel", x[1], "appearing in the image." )