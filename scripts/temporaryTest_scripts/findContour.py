"""
" File:     findContour.py
" Author:   Kevin Ajili <kajili@ucsc.edu>
" Date:     05-18-2018
"""
import numpy as np
import cv2
     
im = cv2.imread('ContaminantSampleOutput/1_3_sec.jpg')

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,0)

im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

cnt = contours[4]
cv2.drawContours(im2, [cnt], 0, (0,255,0), 3)

cv2.imshow("Contours", im2)

while(True):
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()