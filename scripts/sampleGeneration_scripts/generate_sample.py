import sys
import os
import numpy as np
import cv2
import math



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
    


background = cv2.imread(sys.argv[1], 1)
sample_greens = cv2.imread(sys.argv[2], 1)
degrees_per = int(sys.argv[3]) 
times = int(sys.argv[4])

path = './generated_results'

roi = cv2.selectROI(background)
degrees = 0
for i in range(times):
    rotated_sample = rotateImage(sample_greens, degrees)
    combined_image = replace_selection(roi, background, rotated_sample)
    image_name = "sample_image_" + str(i) + ".jpg"
    degrees += degrees_per
    cv2.imwrite(os.path.join(path, image_name), combined_image)

    
