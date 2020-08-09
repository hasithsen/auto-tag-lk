# AutoTag-LK
Automatic License Plate Detection System with Computer Vision for Sri Lanka.

### Guide

#### 1. Execute ```mk_dir_struct.sh``` to create following directory stucture.

```
auto-tag-lk/
  |
  +-- auto_tag.py
  +-- images/
  +-- plates/
  +-- temp/
  +-- footage/
```

#### 2. Place any image/video files inside ```footage``` directory.

#### 3. Use as follows:

###### Images:
  ```
  python3 auto_tag.py -s footage/car_image.png
  ```
  + "car_image.png" would be a image file containing vehicles.

###### Videos:
  ```
  python3 auto_tag.py -source footage/car_video.avi -frequency 15
  ```
  + -source \| -s: "car_video.avi" would be a video file containing vehicles.
  + -frequency \| -f: "15" processes a frame every 15 frames.

#### 4. Output will reside in ```temp``` directory.

### To-do

- [x] Detect white plates
- [ ] Detect yellow plates
- [x] Recognize plates (accuracy varies)
- [ ] Web interface
- [x] Recognize from video/image files

### Contributing

Contributions to AutoTag-LK are welcome. The most helpful way is to try it out and give feedback. Feel free to use the Github issue tracker and pull requests to discuss and submit code changes.
