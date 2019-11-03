#!/usr/bin/env bash

vid_file="${1}"
capture_fps="${2}"
rm -rf images/ plates/ temp/
mkdir -p images/ plates/ temp/
python3 extract_vid.py "${vid_file}" "${capture_fps}"
sleep 1
echo "Resizing images ..."
i=1; for e in $(ls -1 images); do convert -resize 60% images/${e} images/${e}; ((i=i+1)); done # rescaling integrated to auto-tag.py
./run_batch.sh auto_tag.py
echo "Generating video from images ..."
python3 make_vid.py temp

