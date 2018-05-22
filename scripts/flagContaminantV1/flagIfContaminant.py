"""
" File:     flagIfContaminant.py
" Author:   Kevin Ajili <kajili@ucsc.edu>
" Author:   An Tran <anngtran@ucsc.edu>
" Date:     05-21-2018
"""

import time
import argparse
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

startTime = time.time()

#usage python3 flagIfContaminant.py -i example.jpg
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# Apply Gaussian Thresholding to image
imgBeforeThresh = cv2.imread(args["image"],0)
imgBeforeThresh = cv2.medianBlur(imgBeforeThresh,5)
imgGauss = cv2.adaptiveThreshold(imgBeforeThresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

# Convert imgGauss array into image for use with PIL libraries
image = Image.fromarray(imgGauss)

# Save image after applying Gaussian Thresholding
image.save('gaussThresh.jpg')

#getcolors() gets a list of rgb in an image. return a list of (count, pixel)
colors = image.convert('RGB').getcolors(100000)

sortedColors = sorted(colors)

#Percent from the number of times a certain color pixel appears over all the pixels in an image
totalPixels = 0
whitePixels = 0
for x in sortedColors:
    totalPixels = totalPixels + x[0] 
    if(235 < x[1][0] and 235 < x[1][1] and 235 < x[1][2]):  
        whitePixels = whitePixels + x[0]

percentOfWhite = (whitePixels / totalPixels) * 100

# Print the percentage of white pixels that appears in the image
print(str(percentOfWhite) + "% of white pixels in the image." )

contaminantIsFound = False

# If image has less than 99.78% white pixels, it's very likely to contain a contaminant.
#  Note: This is based on testing images after masking green + background,
#  where 99.80% is the lowest value found in images with no contaminants. Subject to change. 
if(percentOfWhite < 99.78):
    contaminantIsFound = True

print("Contaminant Found: " + str(contaminantIsFound))

# Displays running time
print("Running Time: " + str(time.time() - startTime) + " seconds")