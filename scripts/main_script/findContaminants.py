#!/usr/env/bin python3

"""
" File:     findContaminants.py
" Date:     05-21-2018
"""

import time
import os
import cv2
import utils
import argparse
import tkinter as tk
import multiprocessing as mp
from matplotlib import pyplot as plt

SLEEP_TIME = 2
DEFAULT_PIXEL_SIZE = 10000
DEFAULT_WHITE_PERCENTAGE = 99.4
DEFAULT_OUTPUT_DIRECTORY = "ContaminantOutput"


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
    #cv2.namedWindow('highlight_contaminant', cv2.WINDOW_NORMAL)

    cv2.moveWindow("original", 0, 30);
    cv2.moveWindow("background_strip", mid_width, 30);
    cv2.moveWindow("green_strip", mid_width*2, 30);
    cv2.moveWindow("flag_contaminant", mid_width, mid_height+50);
    #cv2.moveWindow("highlight_contaminant", mid_width, mid_height+50);
    #cv2.moveWindow("contaminants", mid_width*2, mid_height+50);

    cv2.resizeWindow('original', mid_width, mid_height)
    cv2.resizeWindow('background_strip', mid_width, mid_height)
    cv2.resizeWindow('green_strip', mid_width, mid_height)
    cv2.resizeWindow('flag_contaminant', mid_width, mid_height)
    #cv2.resizeWindow('highlight_contaminant', mid_width, mid_height)
    #cv2.resizeWindow('contaminants', mid_width, mid_height)

def processDisplay(original, backgroundStrip, greenStrip, flagContam): #, highlContam):
    """
    Updates and displays ORIGINAL, BACKGROUNDSTRIP and GREENSTRIP images/frames
    in their corresponding windows
    """
    cv2.imshow('original', original)
    cv2.imshow('background_strip', backgroundStrip)           
    cv2.imshow('green_strip', greenStrip)
    cv2.imshow('flag_contaminant', flagContam)
    #cv2.imshow('highlight_contaminant', highlContam)

def createOutputDirectory(directoryPath):
    """
    Create directory at DIRECTORYPATH if it doesn't exist already
    """
    if not os.path.exists(directoryPath):                       
        os.makedirs(directoryPath)

def processImageHighlight(imgQueue, outputDirPath, minPixelSize):
    """
    Function to be used as a separate process that highlight an image if one exists in the queue,
    otherwise, process sleeps
    """

    while True:

        if imgQueue.empty():
            #Sleep 2 seconds
            time.sleep(SLEEP_TIME)

        else:
            workingImg, timeSt = imgQueue.get()

            #Check to highlight contaminant
            highlightImg, isFoundSecondPass = utils.highlightContaminant(workingImg, minPixelSize)

            #If highlight works
            if isFoundSecondPass:           
                cv2.imwrite(outputDirPath + '/' +  str(timeSt) + ".jpg", highlightImg)


def main():

    #Arguments
    videoPath, outputDirPath, bckgdMaskPath, longFlag, whitePercentFlag, minPixelSize = parseArguments()
    if bckgdMaskPath == None:
        bckgdMaskPath = utils.createBackgroundMask(videoPath)

    #Initialize objects
    createOutputDirectory(outputDirPath)
    bMask = cv2.imread(bckgdMaskPath)
    videoCapture = cv2.VideoCapture(videoPath)

    #Initialize queue and child process
    ImageQueue = mp.Queue()
    highlightingProcess = mp.Process(target=processImageHighlight, args=(ImageQueue, outputDirPath, minPixelSize) )
    highlightingProcess.start()

    if longFlag:
        resizeWindows()

    #Go trough every frame
    while True:
        ret, frame = videoCapture.read()

        #Video is over
        if ret == False:
            break

        #Mask the background
        backgroundStrip = utils.maskBackground(bMask, frame)
        #Mask color green
        greensOnly, greenStrip = utils.maskGreen(backgroundStrip)
        #Check for possible contaminant
        contaminantImg, isFoundFirstPass = utils.findContaminant(greenStrip, whitePercentFlag)

        #If image is flagged
        if isFoundFirstPass:

            #Timestamps is passed along with image
            timestamp = utils.getTimeStamp(videoCapture)

            ImageQueue.put( (greenStrip, timestamp) )

        #Display different masked frames
        if longFlag:
            processDisplay(frame, backgroundStrip, greenStrip, contaminantImg ) 

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    videoCapture.release()
    cv2.destroyAllWindows()

    #Display different masked frames
    if longFlag:
        print("Video feed over.")

    print("Waiting for image processing to complete...")

    #Allow child process to complete before terminating
    while True:

        if ImageQueue.empty():
            #Allow an extra 3 seconds for last image to be created
            time.sleep(3)

            highlightingProcess.terminate()
            highlightingProcess.join()
            break

        else:    
            #Sleep 2 seconds
            time.sleep(SLEEP_TIME)

    print("Image processing complete.")

    
if __name__ == '__main__':
    main()

