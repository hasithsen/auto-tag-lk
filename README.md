# auto-tag-lk
Automatic License Plate Detection System with Computer Vision for Sri Lanka

###Guide

##1. create folder stucture as follows

"auto-tag" folder
  |
  +-- auto_tag.py
  +-- "images" (empty folder)
  +-- "plates" (empty folder)
  +-- "temp" (empty folder)
  +-- "footage" (images/videos)

##2. Place any image/video files inside "footage" folder.

##3. Use script as follows:

  run script as "python3 auto_tag.py -s footage/car_image.png"
    + "car_image.png" is the image file containing a car with number plate

  run script as "python3 auto_tag.py -s footage/car_video.avi -f 15"
    + -s: "car_video.avi" is the video file containing a car with number plate
    + -f: "15" is the rate to capture frames from video

##4. Output will be in "temp folder".

##5. Change code as you wish. Try to understand the script.