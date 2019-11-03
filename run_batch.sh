#!/usr/bin/env bash

# Run a command in batch mode

if [[ ${1} == "" ]]; then
  echo "Please provide a script name."
  exit 0
fi

cmd="${1}"
img_dir="images"
images="$(ls -1 ${img_dir} | grep '.png\|.PNG\|.jpg\|.JPG')"
set -v # enable verbose execution
for image in ${images}; do
  python3 "${1}" -s "${img_dir}/${image}"
done
