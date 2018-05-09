#Note: colors in openCV are in GBR order instead of RGB
#Usage: python ./colorDetect.py --image [image name]

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#green boundaries in HSV
lower_bound = np.array([35, 5, 5])
upper_bound = np.array([79, 255, 255])

#for (lower, upper) in boundaries:
    #lower = np.array(lower, dtype = "uint8")
    #upper = np.array(upper, dtype = "uint8")

mask = cv2.inRange(hsv, lower_bound, upper_bound)

#show image with mask applied
output = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Greens", np.hstack([image, output]))

#show image with inverted mask
mask2 = cv2.bitwise_not(mask)
inverted_output = cv2.bitwise_and(image, image, mask = mask2)
cv2.imshow("Not_greens", np.hstack([image, inverted_output]))


cv2.waitKey(0)