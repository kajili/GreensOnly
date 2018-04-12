'''
purpose: extracts every frame of a video file and converts them to .png

requires openCV: pip install opencv-python

Note: .png files are saved to the same folder
'''

import cv2
vidcap = cv2.VideoCapture('test.mp4')  # <----put video filename here
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file      
  success,image = vidcap.read()
  print('Frame saved: ' + str(count) + ' ', success)
  count += 1