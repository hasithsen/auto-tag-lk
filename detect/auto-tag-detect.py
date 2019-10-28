#!/usr/bin/env python3

from TOOLS import Functions

import cv2
import numpy as np
import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to image")
ap.add_argument("-o", "--output-dir", type=str, default="plates", help="directory to save output")
args = vars(ap.parse_args())

# this folder is used to save the image
temp_folder = args["output_dir"]+"/"
img = cv2.imread(args["image"])

import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess
import argparse

def preprocess(img):
  cv2.imshow("Input",img)
  imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
  gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

  sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
  #cv2.imshow("Sobel",sobelx)
  #cv2.waitKey(0)
  ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #cv2.imshow("Threshold",threshold_img)
  #cv2.waitKey(0)
  return threshold_img

def cleanPlate(plate):
  print("CLEANING PLATE. . .")
  gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
  #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  #thresh= cv2.dilate(gray, kernel, iterations=1)

  _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
  contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  if contours:
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)

    max_cnt = contours[max_index]
    max_cntArea = areas[max_index]
    x,y,w,h = cv2.boundingRect(max_cnt)

    if not ratioCheck(max_cntArea,w,h):
      return plate,None

    cleaned_final = thresh[y:y+h, x:x+w]
    #cv2.imshow("Function Test",cleaned_final)
    return cleaned_final,[x,y,w,h]

  else:
    return plate,None


def extract_contours(threshold_img):
  element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
  morph_img_threshold = threshold_img.copy()
  cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
  cv2.imshow("Morphed",morph_img_threshold)
  cv2.waitKey(0)

  contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
  return contours


def ratioCheck(area, width, height):
  ratio = float(width) / float(height)
  if ratio < 1:
    ratio = 1 / ratio

  aspect = 4.7272
  min = 15*aspect*15  # minimum area
  max = 125*aspect*125  # maximum area

  rmin = 3
  rmax = 6

  if (area < min or area > max) or (ratio < rmin or ratio > rmax):
    return False
  return True

def isMaxWhite(plate):
  avg = np.mean(plate)
  if(avg>=115):
    return True
  else:
     return False

def validateRotationAndRatio(rect):
  (x, y), (width, height), rect_angle = rect

  if(width>height):
    angle = -rect_angle
  else:
    angle = 90 + rect_angle

  if angle>15:
     return False

  if height == 0 or width == 0:
    return False

  area = height*width
  if not ratioCheck(area,width,height):
    return False
  else:
    return True



def cleanAndRead(img,contours):
  #count=0
  for i,cnt in enumerate(contours):
    min_rect = cv2.minAreaRect(cnt)

    if validateRotationAndRatio(min_rect):

      x,y,w,h = cv2.boundingRect(cnt)
      plate_img = img[y:y+h,x:x+w]


      if(isMaxWhite(plate_img)):
        #count+=1
        clean_plate, rect = cleanPlate(plate_img)

        if rect:
          x1,y1,w1,h1 = rect
          x,y,w,h = x+x1,y+y1,w1,h1
          cv2.imshow("Cleaned Plate", clean_plate)
          cv2.imwrite("plate.jpeg", clean_plate)
          cv2.waitKey(0)
          plate_im = Image.fromarray(clean_plate)
          custom_config = r"--oem 3 --psm 6"
          text = tess.image_to_string(plate_im, lang='eng', config=custom_config)
          print("Detected Text : ",text)
          img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
          cv2.imshow("Detected Plate",img)
          cv2.waitKey(0)

  #print "No. of final cont : " , count

# cv2.imshow('original', img)
# cv2.imwrite(temp_folder + '1 - original.png', img)

imgy = cv2.imread(args["image"])
imgy_w, imgy_h = imgy.shape[:2]
imgy_a = imgy_w * imgy_h
#print(imgy_a/50)

#converting frame(imgy) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)
hsv = cv2.cvtColor(imgy, cv2.COLOR_BGR2HSV)

#defining the range of Yellow color
"""
yellow_lower = np.array([22,60,200],np.uint8)
yellow_upper = np.array([60,255,255],np.uint8)"""
yellow_lower = np.array([14,133,67],np.uint8)
yellow_upper = np.array([32,255,255],np.uint8)

#finding the range yellow colour in the image
yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

#Morphological transformation, Dilation         
kernal = np.ones((5 ,5), "uint8")

blue=cv2.dilate(yellow, kernal)

res=cv2.bitwise_and(imgy, imgy, mask = yellow)

#Tracking Colour (Yellow) 
contours,hierarchy = cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

imgy_cropped = None
yellow_plate = False
for pic, contour in enumerate(contours):
  area = cv2.contourArea(contour)
  if(imgy_a/300<area<imgy_a/50):
    #print(area)
    yellow_plate = True           
    x,y,w,h = cv2.boundingRect(contour)     
    padding = 20
    x_ = x-padding/2
    y_ = y-padding/2
    w_ = w+padding
    h_ = h+padding
    plate_center = x_+(w_/2), y_+(h_/2)
    imgy_cropped = cv2.getRectSubPix(imgy, (w_, h_), tuple(plate_center))
    # draw rectangle around plate in original image
    imgy = cv2.rectangle(imgy,(x,y),(x+w,y+h),(0,255,0),2)

if yellow_plate == True:
  cv2.imshow("Color Tracking",imgy)
  #imgy = cv2.flip(imgy,1)
  #cv2.imshow("Yellow",res)
  cv2.imwrite(temp_folder+"plate_"+args["image"], imgy_cropped)
  cv2.imshow("plate", imgy_cropped)
  cv2.waitKey(0)
  exit()

# hsv transform - value = gray image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)
#cv2.imshow('gray', value)
#cv2.imwrite(temp_folder + '2 - gray.png', value)

# kernel to use for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# applying topHat/blackHat operations
topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
#cv2.imshow('topHat', topHat)
#cv2.imshow('blackHat', blackHat)
# cv2.imwrite(temp_folder + '3 - topHat.png', topHat)
# cv2.imwrite(temp_folder + '4 - blackHat.png', blackHat)

# add and subtract between morphological operations
add = cv2.add(value, topHat)
subtract = cv2.subtract(add, blackHat)
#cv2.imshow('subtract', subtract)
# cv2.imwrite(temp_folder + '5 - subtract.png', subtract)

# applying gaussian blur on subtract image
blur = cv2.GaussianBlur(subtract, (5, 5), 0)
#cv2.imshow('blur', blur)
# cv2.imwrite(temp_folder + '6 - blur.png', blur)

# thresholding
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
#cv2.imshow('thresh', thresh)
# cv2.imwrite(temp_folder + '7 - thresh.png', thresh)

# check for contours on thresh
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# get height and width
height, width = thresh.shape

# create a numpy array with shape given by threshed image value dimensions
imageContours = np.zeros((height, width, 3), dtype=np.uint8)
# list and counter of possible chars
possibleChars = []
countOfPossibleChars = 0

# loop to check if any (possible) char is found
for i in range(0, len(contours)):
  # draw contours based on actual found contours of thresh image
  cv2.drawContours(imageContours, contours, i, (255, 255, 255))
  
  # detect canny edges
  canny = cv2.Canny(imageContours, 5, 100)
  #cv2.imshow("canny", canny)
  
  # retrieve a possible char by the result ifChar class give us
  possibleChar = Functions.ifChar(contours[i])

  # by computing some values (area, width, height, aspect ratio) possibleChars list is being populated
  if Functions.checkIfChar(possibleChar) is True:
    countOfPossibleChars = countOfPossibleChars + 1
    possibleChars.append(possibleChar)

  #cv2.imshow("contours", imageContours)
  #cv2.imwrite(temp_folder + '8 - imageContours.png', imageContours)

imageContours = np.zeros((height, width, 3), np.uint8)

ctrs = []

# populating ctrs list with each char of possibleChars
for char in possibleChars:
  ctrs.append(char.contour)

# using values from ctrs to draw new contours
cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))
#cv2.imshow("contoursPossibleChars", imageContours)
#cv2.imwrite(temp_folder + '9 - contoursPossibleChars.png', imageContours)

plates_list = []
listOfListsOfMatchingChars = []

for possibleC in possibleChars:

  # the purpose of this function is, given a possible char and a big list of possible chars,
  # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
  def matchingChars(possibleC, possibleChars):
    listOfMatchingChars = []

    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
    # then we should not include it in the list of matches b/c that would end up double including the current char
    # so do not add to list of matches and jump back to top of for loop
    for possibleMatchingChar in possibleChars:
      if possibleMatchingChar == possibleC:
        continue

      # compute stuff to see if chars are a match
      distanceBetweenChars = Functions.distanceBetweenChars(possibleC, possibleMatchingChar)

      angleBetweenChars = Functions.angleBetweenChars(possibleC, possibleMatchingChar)

      changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
        possibleC.boundingRectArea)

      changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
        possibleC.boundingRectWidth)

      changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
        possibleC.boundingRectHeight)

      # check if chars match
      if distanceBetweenChars < (possibleC.diagonalSize * 5) and \
          angleBetweenChars < 12.0 and \
          changeInArea < 0.5 and \
          changeInWidth < 0.8 and \
          changeInHeight < 0.2:
        listOfMatchingChars.append(possibleMatchingChar)

    return listOfMatchingChars


  # here we are re-arranging the one big list of chars into a list of lists of matching chars
  # the chars that are not found to be in a group of matches do not need to be considered further
  listOfMatchingChars = matchingChars(possibleC, possibleChars)

  listOfMatchingChars.append(possibleC)

  # if current possible list of matching chars is not long enough to constitute a possible plate
  # jump back to the top of the for loop and try again with next char
  if len(listOfMatchingChars) < 3:
    continue

  # here the current list passed test as a "group" or "cluster" of matching chars
  listOfListsOfMatchingChars.append(listOfMatchingChars)

  # remove the current list of matching chars from the big list so we don't use those same chars twice,
  # make sure to make a new big list for this since we don't want to change the original big list
  listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

  recursiveListOfListsOfMatchingChars = []

  for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
    listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

  break

imageContours = np.zeros((height, width, 3), np.uint8)

for listOfMatchingChars in listOfListsOfMatchingChars:
  contoursColor = (255, 0, 255)

  contours = []

  for matchingChar in listOfMatchingChars:
    contours.append(matchingChar.contour)

  cv2.drawContours(imageContours, contours, -1, contoursColor)

  #cv2.imshow("finalContours", imageContours)
  #cv2.imwrite(temp_folder + '10 - finalContours.png', imageContours)

for listOfMatchingChars in listOfListsOfMatchingChars:
  possiblePlate = Functions.PossiblePlate()

  # sort chars from left to right based on x position
  listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

  # calculate the center point of the plate
  plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
  plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0

  plateCenter = plateCenterX, plateCenterY

  # calculate plate width and height
  plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
    len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

  totalOfCharHeights = 0

  for matchingChar in listOfMatchingChars:
    totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

  averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

  plateHeight = int(averageCharHeight * 2)

  # calculate correction angle of plate region
  opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

  hypotenuse = Functions.distanceBetweenChars(listOfMatchingChars[0],
                        listOfMatchingChars[len(listOfMatchingChars) - 1])
  correctionAngleInRad = math.asin(opposite / hypotenuse)
  correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

  # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
  possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

  # get the rotation matrix for our calculated correction angle
  rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

  height, width, numChannels = img.shape

  # rotate the entire image
  imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

  # crop the image/plate detected
  imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))

  # copy the cropped plate image into the applicable member variable of the possible plate
  possiblePlate.Plate = imgCropped

  # populate plates_list with the detected plate
  if possiblePlate.Plate is not None:
    plates_list.append(possiblePlate)

  # draw a ROI on the original image
  for i in range(0, len(plates_list)):
    # finds the four vertices of a rotated rect - it is useful to draw the rectangle.
    p2fRectPoints = cv2.boxPoints(plates_list[i].rrLocationOfPlateInScene)

    # roi rectangle colour
    rectColour = (0, 255, 0)

    cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
    cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
    cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
    cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

    cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
    cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
    cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
    cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

    #cv2.imshow("detected", imageContours)
    #cv2.imwrite(temp_folder + '11 - detected.png', imageContours)

    cv2.imshow("detectedOriginal", img)
    #cv2.imwrite(temp_folder + '12_detected' + args["image"], img)

    cv2.imshow("plate", plates_list[i].Plate)
    cv2.imwrite(temp_folder + 'plate_' + args["image"].split(".")[0] + ".jpg", plates_list[i].Plate)

cv2.waitKey(0)