#!/usr/env/bin python3

"""
" File:     imgpreprocess.py
" Author:   Cesar Neri <ceneri@ucsc.edu>
" Date:     04-13-2018
"""

import os
import sys
import math
from PIL import ImageTk, Image 

TEST_FOLDER_PATH = "./img_data/"
SOURCE_IMG = "test.jpg"
IMAGE_P = TEST_FOLDER_PATH + SOURCE_IMG
MASKED_P = TEST_FOLDER_PATH + "m_" + SOURCE_IMG

class PreprocessedImage:
    """
    PreprocessedImage uses an image to create a matrix object representation of it.
    RGB values can then be obtained for each pixel
    """
    def __init__(self, image_path):
        """
        Constructor takes path to image IMAGE_PATH to initialize object 
        """
        
        self.__filepath = image_path
        self.__image = Image.open(image_path)
        self.__pixels = self.__image.load()

    def getSize(self):
        """
        Returns tuple (width, height) of size of image  
        """
        return self.__image.size

    def setPixel(self, x, y, RGB):
        """
        Sets RGB values of specified pixel  
        """
        self.__pixels[x,y] = RGB
        #self.__image.save(self.__filepath + '_.png')

    def getRGBMatrix(self):
        """
        Returns image matrix as a 2D list
        """
        matrix = []
        width, length = self.getSize()

        for x in range(width):
            row = []
            for y in range(length):

                row.append(self.__pixels[x,y])

            matrix.append(row)

        return matrix

    def printRGBMatrix(self):
        """
        Prints image matrix with RGB values of each pixel. For debugging only.
        """
        width, length = self.getSize()

        for x in range(width):
            for y in range(length):

                print (self.__pixels[x,y], end=" ")

            print()

    def imageAverageColor(self, xLen, yLen):
        """
        Creates a new image of (xLen pixels by yLenpixel ) squares, averaging 
        the pixel values inside the squares
        """
        length, height = self.getSize()

        for i in range(0, length-xLen, xLen):
            for j in range(0, height-yLen, yLen):

                self.gridAverageColor(i, i+xLen, j, j+yLen)

        pathPair = os.path.splitext(self.__filepath)
        masked = pathPair[0] + "_m" + pathPair[1]
        self.__image.save(masked)

    def gridAverageColor(self, x0, x1, y0, y1):
        """
        Simple average of RGB values
        """

        pixelCount = 0
        red, green, blue = 0, 0, 0
        
        for x in range(x0, x1):
            for y in range(y0, y1):

                pixel = self.__pixels[x,y]
                red += pixel[0] 
                green += pixel[1] 
                blue += pixel[2] 
                pixelCount += 1

        avg_red = int(red/pixelCount)
        avg_green = int(green/pixelCount)
        avg_blue = int(blue/pixelCount)

        new_RGB = (avg_red, avg_green, avg_blue)

        for x in range(x0, x1):
            for y in range(y0, y1):

                self.setPixel(x, y, new_RGB)

def averageImage(filepath, pixelLength, pixelHeight):
    """Creates new image that is the result of averaging the image found int FILEPATH 
       and averaging its pixels to PIXELLENGTH by PIXELHEIGHT squares
    """
    workingImg = PreprocessedImage(filepath)
    workingImg.imageAverageColor(pixelLength, pixelHeight)

def main():    

    #test = PreprocessedImage(IMAGE_P)
    #test.imageAverageColor(2, 2)

    #print ("Size", test.getSize())
    #test.printRGBMatrix()
    #print(test.getRGBMatrix())

    filename =  sys.argv[1]
    pixelsPerSquare = int (sys.argv[2])
    workingImg = PreprocessedImage(filename)
    workingImg.imageAverageColor(pixelsPerSquare, pixelsPerSquare)


if __name__ == '__main__':
    main()
