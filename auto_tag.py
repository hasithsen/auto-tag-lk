
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess
import argparse

out_dir = "" # dir to save output
processed_images = [] # array to store processed image frames in video files

def preprocess(img):
  #cv2.imshow("Input",img)
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
  print("Cleaning plate ...")
  gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
  #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  #thresh= cv2.dilate(gray, kernel, iterations=1)

  _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
  contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  #cv2.imshow("Thresh", thresh)
  #cv2.waitKey(0)

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
  #cv2.imshow("Morphed",morph_img_threshold)
  #cv2.waitKey(0)

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



def cleanAndRead(img, contours, img_name, frame_num):
  count=0
  for i,cnt in enumerate(contours):
    min_rect = cv2.minAreaRect(cnt)

    if validateRotationAndRatio(min_rect):

      x,y,w,h = cv2.boundingRect(cnt)
      plate_img = img[y:y+h,x:x+w]


      if(isMaxWhite(plate_img)):
        count+=1
        clean_plate, rect = cleanPlate(plate_img)

        if rect:
          x1,y1,w1,h1 = rect
          x,y,w,h = x+x1,y+y1,w1,h1
          #cv2.imshow("Cleaned Plate",clean_plate)
          #cv2.waitKey(0)
          plate_im = Image.fromarray(clean_plate)
          custom_config = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
          text = tess.image_to_string(plate_im, lang='eng', config=custom_config)
          print("Detected Text : ",text)
          img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
          cv2.imwrite("plates/{0}_plate_{1}_{2}.png".format(img_name, frame_num, count),clean_plate)
  
  global out_dir
  cv2.imwrite(out_dir+"/{0}_img_{1}.png".format(img_name, frame_num),img)
  global processed_images
  processed_images.append(img)
  #cv2.imshow("Detected Plate",img)
  #cv2.waitKey(0)

  #print("No. of final cont : " , count)

def process_img(img, img_name, frame_num):
  print("Detecting plate ...")
  threshold_img = preprocess(img)
  contours= extract_contours(threshold_img)

  tmp = img.copy()
  if len(contours)!=0:
    #print(len(contours)) #Test
    cv2.drawContours(tmp, contours, -1, (0,255,0), 1)
    #cv2.imshow("Contours",tmp)
    #cv2.waitKey(0)

  cleanAndRead(img, contours, img_name, frame_num)

def generate_video():
  global out_dir
  global processed_images

  height,width,layers = processed_images[0].shape
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
  video = cv2.VideoWriter(out_dir+"/"+"output.mp4", fourcc, 6, (width, height))

  for i, image in enumerate(processed_images):
    #print("frame", i)
    video.write(image)
  video.release()

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-s", "--source", type=str, required=True, help="Path to source")
  ap.add_argument("-f", "--frequency", type=int, default=15, help="Process once every n frames (integer)")
  ap.add_argument("-o", "--output-dir", type=str, default="temp", help="Directory to save output")
  args = vars(ap.parse_args())

  global out_dir
  out_dir = args["output_dir"]
  source_filename = ""
  source_dir = ""
  if "/" in args["source"]:
    source_dir = args["source"].rsplit("/")[0]
    source_filename = args["source"].rsplit("/")[1]
  else:
    source_filename = args["source"]

  video_types = ["avi", "mov"]
  image_types = ["png", "jpg"]

  is_vid = False
  for vid_ext in video_types:
    if source_filename.rsplit(".")[1].lower() in vid_ext:
      is_vid = True

  if is_vid:
    vidcap = cv2.VideoCapture(args["source"])
    success,image = vidcap.read()
    frame_num = 0 
    capture_frequency = args["frequency"] # [10, 15] enough for ~24 fps video for near-realtime
    print("Extracting once every {0} frames ...".format(capture_frequency))
    while success:
      if frame_num % capture_frequency == 0:
        print("Processing frame #:", frame_num)
        img = cv2.resize(image, (0,0), fx=0.6, fy=0.6)
        process_img(img, source_filename.rsplit(".")[0], frame_num)
      success,image = vidcap.read()
      #print('Read a new frame: ', success)
      frame_num += 1
    print("Generating video ...")
    generate_video()
    exit()
  else:
    img = cv2.imread(args["source"]) # image
    #img = cv2.resize(image, (0,0), fx=0.6, fy=0.6)
    process_img(img, source_filename.rsplit(".")[0], 0) # frame num zero since single image
    #cv2.imshow(source_filename, processed_images[0])
    #cv2.imwrite(out_dir+"/"+source_filename.rsplit(".")[0]+"_detected.png", processed_images[0]) # single frame since image
    #cv2.waitKey(0)

if __name__ == "__main__":
  main()
