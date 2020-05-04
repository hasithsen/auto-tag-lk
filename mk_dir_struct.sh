#!/usr/bin/env bash

# Create directory structure for auto-tag.py

dir_names=(images plates temp footage)
for e in ${dir_names[@]}; do
  mkdir -p "${e}"
done