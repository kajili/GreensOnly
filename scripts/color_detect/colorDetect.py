#Note: colors in openCV are in GBR order instead of RGB
#Usage: python ./colorDetect.py --image [image name]
#citation: some code from here is used: https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

boundaries = [([0, 60, 0], [100, 255, 100]),
              ([20, 80, 20], [40, 255, 40]),
             ]

for (lower, upper) in boundaries:
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    mask = cv2.inRange(image, lower, upper)
    mask2 = cv2.inRange(image, lower, upper)

    
    for line in mask:
        print(line)
        for element in line:
            print(element)
            
    
    print(mask)
    print(mask2)
    output = cv2.bitwise_and(image, image, mask = mask)

    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)