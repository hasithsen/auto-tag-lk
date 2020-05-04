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

#### 3. Use script as follows:

###### For still image input:
  ```
  python3 auto_tag.py -s footage/car_image.png
  ```
  + "car_image.png" would be a image file containing vehicles.

###### For video footage input:
  ```
  python3 auto_tag.py -s footage/car_video.avi -f 15
  ```
  + -s: "car_video.avi" would be a video file containing vehicles.
  + -f: "15" is video frame capturing frequency (i.e. every 15th frame).

#### 4. Output will reside in ```temp``` directory.

### To-do

- [x] Detect white plates
- [ ] Detect yellow plates
- [x] Recognize plates (accuracy varies)
- [ ] Web interface
- [x] Recognize from video/image files

### Contributing

Contributions to AutoTag-LK are welcome. The most helpful way is to try it out and give feedback. Feel free to use the Github issue tracker and pull requests to discuss and submit code changes.
