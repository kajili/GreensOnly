#!/usr/env/bin python3

"""
" File:     utils.py
" Date:     05-30-2018
"""

import os
import cv2
import shutil
import fnmatch
import numpy as np
from PIL import Image

TEMP_BKGRND_MASK_DIRECTORY =  "./Masks"
TEMP_BKGRND_MASK = "final.jpg"

def turnGreenLight():
    return cv2.imread("green_light_icon.png")

def turnRedLight():
    return cv2.imread("red_light_icon.png")

def getTimeStamp(capture):
    """
    Given CAPTURE object, returns string timestamp of current frame
    """
    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
    timestamp = round(timestamp, 2)
    timestamp = str(timestamp)

    return timestamp.replace('.', '_') + "_sec"

def maskBackground(bMask, frame):
    """
    Applies BMASK (Background Mask) to specified FRAME
    """
    return cv2.bitwise_and(bMask, frame)

def maskGreen(frame):
    """
    Returns the result of masking the color green on FRAME, as well
    as the inverse (Everything but green)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #green boundaries in HSV
    lower_bound = np.array([35, 5, 5])
    upper_bound = np.array([79, 255, 255])

    #Greens only
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    greens = cv2.bitwise_and(frame, frame, mask = mask)

    #Inverted mask
    mask2 = cv2.bitwise_not(mask)
    non_greens = cv2.bitwise_and(frame, frame, mask = mask2)

    return greens, non_greens

def generateIndividualMasks(directoryPath, videoPath):
    """
    Given a short video, MOG masking is applied and resulting mask
    frames are saved to NEW_DIR directory. 
    Return the number of masks generated
    """
    FRAME_NAME = directoryPath + "/mask"
    
    #Create directory if doesnt exist already
    if not os.path.exists(directoryPath):                       
        os.makedirs(directoryPath)

    #Load video
    cap = cv2.VideoCapture(videoPath)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    #counter to skip first couple of frames
    skipImage = 0

    #Go trough every frame
    maskCounter = 0
    while True:
        ret, frame = cap.read()

        #There is another frame
        if ret == True:

            gmask = fgbg.apply(frame)

            #Skip fisrt couple of frames
            if skipImage > 60:

                if skipImage > 210:
                    break
                
                #Images are saved to newDir folder
                cv2.imwrite(FRAME_NAME + str(maskCounter) + ".jpg", gmask)
                maskCounter += 1

            skipImage += 1  
            
        #Video is over
        else:
            break;

    cap.release()
    cv2.destroyAllWindows()

    return maskCounter

def createBlackImage(directoryPath):
    """
    Creates an black image that is of the same size as the images found in 
    the folder at DIRECTORYPATH
    """
    
    #Get path to any image inside the directory
    pattern = "*.jpg"
    for _, _, files in os.walk(directoryPath + '/'):
        sample_path = fnmatch.filter(files, pattern)[0]
        break;

    #Use any image to get right emasurement
    sample_image = cv2.imread(directoryPath + "/" + sample_path)
    height = np.size(sample_image, 0)
    width = np.size(sample_image, 1)

    #Black image of right size
    blk_img = np.zeros((height, width, 3),  np.uint8)
    return blk_img

def combineMasks(directoryPath, numImages):
    """
    Uses masks created by generateIndividualMasks(), to combine them
    and create the final mask 
    """

    IMG_NAME = directoryPath + "/mask"

    mask = createBlackImage(directoryPath)

    for imgCount in range(1,numImages):

        img_path = IMG_NAME + str(imgCount) + ".jpg"
        if os.path.exists(img_path):

            img = cv2.imread(img_path)
            # combine foreground+background
            mask = cv2.bitwise_or(mask, img)

    #Images are saved to newDir folder
    cv2.imwrite(TEMP_BKGRND_MASK, mask)


def createBackgroundMask(videoPath):
    """
    If a mask is not passed a mask for video found in VIDEOPATH is created
    """
    maskCount = generateIndividualMasks(TEMP_BKGRND_MASK_DIRECTORY, videoPath)

    #Delete images if needed during pause/idle time
    input("Press enter when you are ready to combine the masks.")
    
    #Create final mask by combining all individual masks
    combineMasks(TEMP_BKGRND_MASK_DIRECTORY, maskCount)

    shutil.rmtree(TEMP_BKGRND_MASK_DIRECTORY)
    #print("Need to create Mask")
    return TEMP_BKGRND_MASK

def applyThresholding(frame):

    TEMP_IMG_NAME = "temp.jpg"

    image = Image.fromarray(frame)
    # Save image using PIL due to previous errors with adaptiveThreshold only working with imread
    image.save(TEMP_IMG_NAME)

    # Apply Gaussian Thresholding to image
    imgBeforeThresh = cv2.imread("temp.jpg",0)
    imgBeforeThresh = cv2.medianBlur(imgBeforeThresh,5)
    imgGauss = cv2.adaptiveThreshold(imgBeforeThresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    # Remove temporary image used for processing
    os.remove(TEMP_IMG_NAME)
    return imgGauss

def findContaminant(frame, whiteFlag):
    """
    Takes FRAME argument, and returns
    image highlighting contaminants, and boolean if contaminant found
    """

    thresh = applyThresholding(frame)

    # Convert thresh array into image for use with PIL libraries
    image = Image.fromarray(thresh)

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

    contaminantIsFound = False

    # If image has less than 99.40% white pixels, it's very likely to contain a contaminant.
    #  Note: This is based on testing images after masking green + background,
    #  where 99.45% is the lowest value found in images with no contaminants. Subject to change. 
    if(percentOfWhite < whiteFlag):
        # Image is flagged as having a contaminant if it has less than 99.40% white pixels
        contaminantIsFound = True
        # # Print the percentage of white of the passing frames, for testing
        # print(percentOfWhite)

    if contaminantIsFound:
        return turnRedLight(), contaminantIsFound
    else:
        return turnGreenLight(), contaminantIsFound


#******Methods from c_r_d.py*******

# rect = ((min_x, min_y), (max_x, max_y))
def place_bounding_box(image, rect):
    image = cv2.rectangle(image,(rect[0][1],rect[0][0]),(rect[1][1],rect[1][0]),(0,255,0),3)
    return image

def calc_bounding_box(coord_list):
    min_x = min(coord_list, key = lambda t: t[0])[0]
    min_y = min(coord_list, key = lambda t: t[1])[1]
    max_x = max(coord_list, key = lambda t: t[0])[0]
    max_y = max(coord_list, key = lambda t: t[1])[1]
    rect = ((min_x, min_y),(max_x, max_y))

    return rect

def scan_image(image, min_region_size):
    region_list = []

    rows, cols, channels = image.shape
    visited_matrix = [[False for col in range(cols)] for row in range(rows)]
    
    directions = [(-1,-1), (-1, 0), (-1, 1), (0,-1), (0, 1), (1,-1), (1, 0), (1, 1)]

    for row in range(rows):
        for col in range(cols):
            # top level linear scan
            if not visited_matrix[row][col]:
                visited_matrix[row][col] = True
                region = not np.array_equal(image[row,col], [0,0,0])
                # first step of flood fill
                if region:
                    coord_list = []
                    coord_list.append((row,col))
                    to_visit = []
                    for direction in directions:
                        new_loc = (row + direction[0],col + direction[1])
                        if not new_loc[0] < 0 and not new_loc[0] > (rows-1) and not new_loc[1] < 0 and not new_loc[1] > (cols-1):
                            if not visited_matrix[new_loc[0]][new_loc[1]]:
                                to_visit.append(new_loc)
                    #start flood fill
                    while to_visit:
                        current_loc = to_visit.pop()
                        visited_matrix[current_loc[0]][current_loc[1]] = True
                        if not np.array_equal(image[current_loc[0],current_loc[1]], [0,0,0]):
                            coord_list.append(current_loc)
                            for direction in directions:
                                new_loc = (current_loc[0] + direction[0], current_loc[1] + direction[1])
                                if not new_loc[0] < 0 and not new_loc[0] > (rows-1) and not new_loc[1] < 0 and not new_loc[1] > (cols-1):
                                    if not visited_matrix[new_loc[0]][new_loc[1]]:
                                        to_visit.append(new_loc)
                    if len(coord_list) > min_region_size:
                        region_list.append(coord_list)

    return region_list

def highlightContaminant(flaggedImg, minRegionSize):

    #image = cv2.imread(imgPath, 1)
    region_list = scan_image(flaggedImg, minRegionSize)
    if region_list:
        rects = []
        for region in region_list:
            rects.append(calc_bounding_box(region))
        image_with_box = flaggedImg.copy()
        for rect in rects:
            place_bounding_box(image_with_box, rect)
            
        return image_with_box, True

    return turnGreenLight(), False