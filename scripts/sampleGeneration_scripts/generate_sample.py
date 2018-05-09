'''
For usage: python generate_sample.py -h

Selct area to place sample, then press enter.
'''

import sys
import os
import numpy as np
import cv2
import math
import argparse




def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def replace_selection(roi, background, rotated_sample):
    background_template = background.copy()
    cropped_sample = rotated_sample[roi[1]:(roi[1]+roi[3]), roi[0]:(roi[0]+roi[2])].copy()
    background_template[roi[1]:(roi[1]+roi[3]), roi[0]:(roi[0]+roi[2])] = cropped_sample
    return background_template



ap = argparse.ArgumentParser()

ap.add_argument("-b", "--background", help = "base background of image", nargs = 1)
ap.add_argument("-s", "--sample", help = "base sample of image", nargs = 1)
ap.add_argument("-d", "--degrees", help = "minimum number of degrees to rotate", default = [10], type = int, nargs = 1)
ap.add_argument("-q", "--quantity", help = "number of output images", default = [20], type = int, nargs = 1)
ap.add_argument("-p", "--path", help = "output path", default = ["./generated_results"], nargs = 1)
ap.add_argument("-m", "--multiple", help = "flag for allowing multiple region selection on images", nargs = '?', const = 5, default = 0, type = int)

args = vars(ap.parse_args())

background = cv2.imread(args["background"][0], 1)
sample_greens = cv2.imread(args["sample"][0], 1)
degrees_per = args["degrees"][0]
quantity = args["quantity"][0]
path = args["path"][0] 
mul_count = args["multiple"]

roi = cv2.selectROI(background)
degrees = 0
combined_images = []
for i in range(quantity):
    rotated_sample = rotateImage(sample_greens, degrees)
    combined_images.append(replace_selection(roi, background, rotated_sample))
    degrees += degrees_per

for i in range(mul_count):
    roi = cv2.selectROI(combined_images[0])
    degrees = 0
    for index, image in enumerate(combined_images):
        rotated_sample = rotateImage(sample_greens, degrees)
        combined_images[index] = replace_selection(roi, image, rotated_sample)
        degrees += degrees_per


for index, image in enumerate(combined_images):
    image_name = "sample_image_" + str(index) + ".jpg"
    cv2.imwrite(os.path.join(path, image_name), image)
