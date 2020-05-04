import cv2
import sys

# Extract and save image frames from video at given frequency (fps)

img_dir = "images"
vidcap = cv2.VideoCapture(sys.argv[1])
success,image = vidcap.read()
count = 0
capture_fps = int(sys.argv[2]) # [10, 15] is ok for realtime
print("Extracting once every {0} frames ...".format(capture_fps))
while success:
  if count % capture_fps == 0: 
    cv2.imwrite("{0}/frame_{1}.png".format(img_dir, count), image)     # save frame as png file
    #print('Saved frame: ', count)
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1
print("Done.")
