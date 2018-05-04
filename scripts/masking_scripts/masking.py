#!/usr/env/bin python3

"""
" File:     masking.py
" Author:   Cesar Neri  <ceneri@ucsc.edu>
" Author:   Kevin Ajili <kajili@ucsc.edu>
" Author:   An Tran     <anngtran@ucsc.edu>
" Date:     05-2-2018
"""

import os
import cv2
import fnmatch
import argparse
import numpy as np

def parseArguments():
    """
    Parses all terminal arguments passed and returns them in a dictionary 
    """

    parser = argparse.ArgumentParser(description='Mask Creator')

    #parser.add_argument('-m', '--mode', required=True, type=int, 
     #                   help="Execution mode (1 or 2)")
    parser.add_argument('-d', '--directory', required=True, 
                        help="Path to directory where images created should be stored")
    parser.add_argument('-v', '--video', required=True,
                        help="Path to video for mask to be based of")

    args = vars(parser.parse_args())
    return args

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

    #Go trough every frame
    maskCounter = 0
    while True:
        ret, frame = cap.read()

        #There is another frame
        if ret == True:

            gmask = fgbg.apply(frame)

            cv2.imshow('original',frame)
            cv2.imshow('mask',gmask)

            #Images are saved to newDir folder
            cv2.imwrite(FRAME_NAME + str(maskCounter) + ".jpg", gmask)
            maskCounter += 1

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

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
    cv2.imwrite(directoryPath + "/final.jpg", mask)

def test():
    """
    Test in a single image (For debugging only)
    """
    #Create two simple images
    img = cv2.imread("img_data/lol.jpg")
    mask = cv2.imread("Masks/final.jpg")
    
    #Code to invert colors
    #mask = (255-mask)

    #Mask
    result = cv2.bitwise_and(mask, img)
    
    #Needed to show image until you enter ESC
    while True:
        cv2.imshow('result', result)

        k = cv2.waitKey(5000) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()

def maskVideo(directoryPath, videoPath):
    """
    Applies the mask to every frame in the original video. Displays the result.
    takes DIRECTORYPATH to get the final mask image, and the VIDEOPATH of the original
    video
    """

    #Load final mask
    mask = cv2.imread(directoryPath + "/final.jpg")

    #Load video
    cap = cv2.VideoCapture(videoPath)

    #Go through every frame
    while True:
        ret, frame = cap.read()

        #There is another frame
        if ret == True:

            result = cv2.bitwise_and(mask, frame)

            cv2.imshow('Result', result)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        #Video is over
        else:
            break;

    cap.release()
    cv2.destroyAllWindows()


def main():

    args = parseArguments()
    DIR_PATH = args['directory']
    VID_PATH = args['video']
  
    individual_masks = generateIndividualMasks(DIR_PATH, VID_PATH)

    #Delete images if needed during pause/idle time
    input("Press enter when you are ready to combine the masks.")

    #Create final mask by combining all individual masks
    combineMasks(DIR_PATH, individual_masks)

    #Display original video with mask applied
    maskVideo(DIR_PATH, VID_PATH)

  
if __name__ == '__main__':
  main()
