"""
" File:     applyMask.py
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

    parser = argparse.ArgumentParser(description='Mask Application Script')

    #parser.add_argument('-m', '--mode', required=True, type=int, 
     #                   help="Execution mode (1 or 2)")
    parser.add_argument('-m', '--mask', required=True, 
                        help="Path to the final mask to be applied.")
    parser.add_argument('-v', '--video', required=True,
                        help="Path to video that mask will be applied to.")

    args = vars(parser.parse_args())
    return args


def maskVideo(maskPath, videoPath):
    """
    Applies the mask to every frame in the original video. Displays the result.
    takes MASKPATH to get the final mask image, and the VIDEOPATH of the original
    video
    """

    #Load final mask
    mask = cv2.imread(maskPath)

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
	MASK_PATH = args['mask']
	VID_PATH = args['video']

	#Display video with mask applied
	maskVideo(MASK_PATH, VID_PATH)

  
if __name__ == '__main__':
  main()