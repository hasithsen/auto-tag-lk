#!/usr/bin/env python3

# Create and save a video by combining multiple image file.

import cv2
import numpy as np
import glob
import sys

img_array = []
for filename in glob.glob(sys.argv[1]+"/*.png"):
  img = cv2.imread(filename)
  img_array.append(img)

  height, width, _ = img_array[0].shape # ignore layers
  size = (width,height)
  #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','P','E','G'), 10, (width, height))
  out = cv2.VideoWriter("detected.avi",cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
  out.write(img_array[i])
  out.release()
