#!/usr/bin/env bash

cmd="${1}"
for image in $(ls *.jpg); do
  ./"${1}" -i $image
done