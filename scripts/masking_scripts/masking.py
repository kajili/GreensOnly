#!/usr/env/bin python3

"""
" File:     masking.py
" Author:   Cesar Neri  <ceneri@ucsc.edu>
" Author:   Kevin Ajili <kajili@ucsc.edu>
" Author:   An Tran <anngtran@ucsc.edu>
" Date:     05-2-2018
"""

import os
import cv2
import numpy as np

NEW_DIR = "./Masks/"

def removeBackgroungMOG():
  """
  Given a short video, MOG masking is applied and resulting mask
  frames are saved to NEW_DIR directory
  """

  cap = cv2.VideoCapture('coins.mp4')

  fgbg = cv2.createBackgroundSubtractorMOG2()

  ## Checks if the path already exists 
  if not os.path.exists(NEW_DIR):  
    ## Creates a new directory with the names of source and template images                        
    os.makedirs(NEW_DIR)

  maskCounter = 0
  while True:
    et, frame = cap.read()
    gmask = fgbg.apply(frame)

    cv2.imshow('original',frame)
    cv2.imshow('mask',gmask)

    #Images are saved to newDir folder
    cv2.imwrite(NEW_DIR + "mask" + str(maskCounter) + ".jpg", gmask)
    maskCounter += 1

    k = cv2.waitKey(30) & 0xff
    if k == 27:
      break
   
  cap.release()
  cv2.destroyAllWindows()

def combineMasks(numImages, width, height):
  """
  Uses masks created by removeBackgroungMOG(), to combine them
  and create the final mask 
  """

  mask = cv2.imread(NEW_DIR + "mask0.jpg")

  for imgCount in range(1,numImages):

    img_path = NEW_DIR + "mask" + str(imgCount) + ".jpg"
    if os.path.exists(img_path):

      img = cv2.imread(img_path)
      # combine foreground+background
      mask = cv2.bitwise_or(mask, img)

      print(imgCount)

  #Images are saved to newDir folder
  cv2.imwrite(NEW_DIR + "final.jpg", mask)

def main():

  #removeBackgroungMOG()

  combineMasks(366, 1280, 720)

if __name__ == '__main__':
  main()