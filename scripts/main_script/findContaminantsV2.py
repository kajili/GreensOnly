#!/usr/env/bin python3

"""
" File:     findContaminants.py
" Date:     05-10-2018
"""

import os
import cv2
import random ##!!For now
#import fnmatch
import argparse
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from PIL import Image



def parseArguments():
    """
    Parses all terminal arguments needed and returns them in a dictionary 
    """

    parser = argparse.ArgumentParser(description='Contaminant Detection')

    parser.add_argument('-v', '--video', required=True,
                        help="Path to video.")
    parser.add_argument('-o', '--directory', required=False, default="ContaminantOutput",
                        help="Path to directory where output images created should be stored.")
    parser.add_argument('-b', '--bMask', required=False, default=None,
                        help="Path to background mask (image).")
    parser.add_argument('-l', '--long', required=False, default=False, action='store_true', 
                        help="Long mode shows masking of every frame as it goes.")
    
    args = vars(parser.parse_args())

    videoP = args['video']
    outputDir = args['directory']
    backgroundMask = args['bMask']
    longFlag = args['long']
    return videoP, outputDir, backgroundMask, longFlag 

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
    cv2.namedWindow('green_only', cv2.WINDOW_NORMAL)
    cv2.namedWindow('color_palette', cv2.WINDOW_NORMAL)
    cv2.namedWindow('contaminants', cv2.WINDOW_NORMAL)

    cv2.moveWindow("original", 0, 30);
    cv2.moveWindow("background_strip", mid_width, 30);
    cv2.moveWindow("green_strip", mid_width*2, 30);
    cv2.moveWindow("green_only", 0, mid_height+30);
    cv2.moveWindow("color_palette", mid_width, mid_height+30);
    cv2.moveWindow("contaminants", mid_width*2, mid_height+30);

    cv2.resizeWindow('original', mid_width, mid_height)
    cv2.resizeWindow('background_strip', mid_width, mid_height)
    cv2.resizeWindow('green_strip', mid_width, mid_height)
    cv2.resizeWindow('green_only', mid_width, mid_height)
    cv2.resizeWindow('color_palette', mid_width, mid_height)
    cv2.resizeWindow('contaminants', mid_width, mid_height)

def processDisplay(original, backgroundStrip, greenStrip, greenOnly):
    """
    Updates and displays ORIGINAL, BACKGROUNDSTRIP and GREENSTRIP images/frames
    in their corresponding windows
    """
    cv2.imshow('original', original)
    cv2.imshow('background_strip', backgroundStrip)           
    cv2.imshow('green_strip', greenStrip)
    cv2.imshow('green_only', greenOnly) 

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

def createBackgroundMask():
    """
    !!TEMPORARY FUNCTION
    If a mask is not passed a mask should be created, for now a mask 
    should ***ALWAYS BE PASSED*** otherwise, may god have mercy on you!
    """
    print("Need to create Mask")
    return "lel"

def dummyColorPalette(frame):
    """
    !!TEMPORARY FUNCTION
    Should return image with fixed color palette
    """
    return frame

def applyThresholding(frame):

    image = Image.fromarray(frame)
    # Save image using PIL due to previous errors with adaptiveThreshold only working with imread
    image.save("temp.jpg")

    # Apply Gaussian Thresholding to image
    imgBeforeThresh = cv2.imread("temp.jpg",0)
    imgBeforeThresh = cv2.medianBlur(imgBeforeThresh,5)
    imgGauss = cv2.adaptiveThreshold(imgBeforeThresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    return imgGauss

def findContaminant(frame):
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
    if(percentOfWhite < 99.4):
        # Image is flagged as having a contaminant if it has less than 99.40% white pixels
        contaminantIsFound = True
        # # Print the percentage of white of the passing frames, for testing
        # print(percentOfWhite)

    return frame, contaminantIsFound

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
    videoPath, outputDirPath, bckgdMaskPath, longFlag = parseArguments()
    if bckgdMaskPath == None:
        bckgdMaskPath = createBackgroundMask()

    #Initialize objects
    createOutputDirectory(outputDirPath)
    bMask = cv2.imread(bckgdMaskPath)
    videoCapture = cv2.VideoCapture(videoPath)

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
        #!!Simplify color palette
        colorPalette = dummyColorPalette(greenStrip)
        #Identify large chunks of the same color
        contaminantImg, isFound = findContaminant(greenStrip)

        #If image is flagged, save img with timestamp
        if isFound:
            timestamp = getTimeStamp(videoCapture)
            cv2.imwrite(outputDirPath + '/' +  str(timestamp) + ".jpg", contaminantImg)


        #Display different masked frames
        if longFlag:
            processDisplay(frame, backgroundStrip, greenStrip, greensOnly)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    videoCapture.release()
    cv2.destroyAllWindows()
    # Remove temporary image used for processing
    os.remove("temp.jpg")



if __name__ == '__main__':
    main()
