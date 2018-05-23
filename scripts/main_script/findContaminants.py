#!/usr/env/bin python3

"""
" File:     findContaminantsV2.py
" Date:     05-21-2018
"""

import os
import cv2
import argparse
import numpy as np
import tkinter as tk
from PIL import Image
from matplotlib import pyplot as plt
import fnmatch
import shutil

DEFAULT_WHITE_PERCENTAGE = 99.4
DEFAULT_PIXEL_SIZE = 10000
DEFAULT_OUTPUT_DIRECTORY = "ContaminantOutput"

def turnGreenLight():
    return cv2.imread("green_light_icon.png")

def turnRedLight():
    return cv2.imread("red_light_icon.png")

def parseArguments():
    """
    Parses all terminal arguments needed and returns them in a dictionary 
    """

    parser = argparse.ArgumentParser(description='Contaminant Detection')

    parser.add_argument('-v', '--video', required=True,
                        help="Path to video.")
    parser.add_argument('-o', '--directory', required=False, default=DEFAULT_OUTPUT_DIRECTORY,
                        help="Path to directory where output images created should be stored.")
    parser.add_argument('-b', '--bMask', required=False, default=None,
                        help="Path to background mask (image).")
    parser.add_argument('-w', '--white', required=False, default=DEFAULT_WHITE_PERCENTAGE, type=float,
                        help="Percent value of white to be used when evaluating potential for contaminant.")
    parser.add_argument('-p', '--pixels', required=False, default=DEFAULT_PIXEL_SIZE, type=int,
                        help="""Minimum pixel size of a region that should be identified as contaminant, 
                        and highlighted inside rectangular box.""")
    parser.add_argument('-l', '--long', required=False, default=False, action='store_true', 
                        help="Long mode shows masking of every frame as it goes.")
    
    args = vars(parser.parse_args())

    videoP = args['video']
    outputDir = args['directory']
    backgroundMask = args['bMask']
    longFlag = args['long']
    whiteFlag = args['white']
    minPixls = args['pixels']
    return videoP, outputDir, backgroundMask, longFlag, whiteFlag, minPixls

def resizeWindows():
    """
    Resizes and alligns all windows to amke the all visible.
    Size and position are based on current monitors specifications
    """

    #Obtain screen size
    #https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    mid_width = screen_width//3
    mid_height = screen_height//3

    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('background_strip', cv2.WINDOW_NORMAL)
    cv2.namedWindow('green_strip', cv2.WINDOW_NORMAL)
    cv2.namedWindow('flag_contaminant', cv2.WINDOW_NORMAL)
    cv2.namedWindow('highlight_contaminant', cv2.WINDOW_NORMAL)

    cv2.moveWindow("original", 0, 30);
    cv2.moveWindow("background_strip", mid_width, 30);
    cv2.moveWindow("green_strip", mid_width*2, 30);
    cv2.moveWindow("flag_contaminant", 0, mid_height+30);
    cv2.moveWindow("highlight_contaminant", mid_width, mid_height+30);
    #cv2.moveWindow("contaminants", mid_width*2, mid_height+30);

    cv2.resizeWindow('original', mid_width, mid_height)
    cv2.resizeWindow('background_strip', mid_width, mid_height)
    cv2.resizeWindow('green_strip', mid_width, mid_height)
    cv2.resizeWindow('flag_contaminant', mid_width, mid_height)
    cv2.resizeWindow('highlight_contaminant', mid_width, mid_height)
    #cv2.resizeWindow('contaminants', mid_width, mid_height)

def processDisplay(original, backgroundStrip, greenStrip, flagContam, highlContam):
    """
    Updates and displays ORIGINAL, BACKGROUNDSTRIP and GREENSTRIP images/frames
    in their corresponding windows
    """
    cv2.imshow('original', original)
    cv2.imshow('background_strip', backgroundStrip)           
    cv2.imshow('green_strip', greenStrip)
    cv2.imshow('flag_contaminant', flagContam)
    cv2.imshow('highlight_contaminant', highlContam)

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

def createOutputDirectory(directoryPath):
    """
    Create directory at DIRECTORYPATH if it doesn't exist already
    """
    if not os.path.exists(directoryPath):                       
        os.makedirs(directoryPath)

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

    skipImage = 0

    #Go trough every frame
    maskCounter = 0
    while True:
        ret, frame = cap.read()

        #There is another frame
        if ret == True:

            gmask = fgbg.apply(frame)

            if skipImage > 60 and skipImage < 210:
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

            #print(imgCount)

    #Images are saved to newDir folder
    cv2.imwrite("final.jpg", mask)


def createBackgroundMask(videoPath):
    """
    !!TEMPORARY FUNCTION
    If a mask is not passed a mask should be created, for now a mask 
    should ***ALWAYS BE PASSED*** otherwise, may god have mercy on you!
    """
    Mask_directory = "./Mask" 
    maskcount = generateIndividualMasks(Mask_directory, videoPath)
    input("Press enter when you are ready to combine the masks.")
    combineMasks(Mask_directory, maskcount)

    shutil.rmtree(Mask_directory)
    #print("Need to create Mask")
    return "final.jpg"

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

def getTimeStamp(capture):
    """
    Given CAPTURE object, returns string timestamp of current frame
    """
    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
    timestamp = round(timestamp, 2)
    timestamp = str(timestamp)

    return timestamp.replace('.', '_') + "_sec"


def main():

    #Arguments
    videoPath, outputDirPath, bckgdMaskPath, longFlag, whitePercentFlag, minPixelSize = parseArguments()
    if bckgdMaskPath == None:
        bckgdMaskPath = createBackgroundMask(videoPath)

    #Initialize objects
    createOutputDirectory(outputDirPath)
    bMask = cv2.imread(bckgdMaskPath)
    videoCapture = cv2.VideoCapture(videoPath)

    #Image declaration, for faster calling
    greenLight = turnGreenLight()

    if longFlag:
        resizeWindows()

    #Go trough every frame
    while True:
        ret, frame = videoCapture.read()

        #Video is over
        if ret == False:
            break

        #Mask the background
        backgroundStrip = maskBackground(bMask, frame)
        #Mask color green
        greensOnly, greenStrip = maskGreen(backgroundStrip)
        #Check for possible contaminant
        contaminantImg, isFoundFirstPass = findContaminant(greenStrip, whitePercentFlag)
        #
        highlightImg = greenLight

        #If image is flagged
        if isFoundFirstPass:

            #Check to highlight contaminant
            highlightImg, isFoundSecondPass = highlightContaminant(greenStrip, minPixelSize)

            #If highlight works
            if isFoundSecondPass:

                timestamp = getTimeStamp(videoCapture)
                cv2.imwrite(outputDirPath + '/' +  str(timestamp) + ".jpg", highlightImg)

        #Display different masked frames
        if longFlag:
            processDisplay(frame, backgroundStrip, greenStrip, contaminantImg, highlightImg)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    videoCapture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
