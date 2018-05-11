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

def dummyContaminantImage(frame, dummyCount):
    """
    !!TEMPORARY FUNCTION
    Real function should only take FRAME argument, and return 
    image highlighting contaminants, and boolean if contaminant found
    """
    contaminantFrames = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 100]

    return frame, dummyCount in contaminantFrames

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

    #!!FOR NOW ONLY!!!
    dummyCount = 0

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
        #!!Identify large chunks of the same color
        contaminantImg, isFound = dummyContaminantImage(colorPalette, dummyCount)

        #If image is flagged, save img with timestamp
        if isFound:
            timestamp = getTimeStamp(videoCapture)
            cv2.imwrite(outputDirPath + '/' +  str(timestamp) + ".jpg", contaminantImg)

        #!!FOR NOW ONLY!!!
        dummyCount += 1

        #Display different masked frames
        if longFlag:
            processDisplay(frame, backgroundStrip, greenStrip, greensOnly)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    videoCapture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
