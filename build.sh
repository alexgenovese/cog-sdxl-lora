#!/bin/bash

set -o errexit
set -o xtrace

echo "Build the base + weights"

BASE_IMAGE="r8.im/alexgenovese/sdxl-lora"

# echo "ignore all the small files, copy big files"
# find . -type f -size -10M > .dockerignore
# docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t base .

# BASE_ID=$(docker inspect $(BASE_IMAGE) --format='{{index .Id}}')

echo "ignore all the big files, copy copy files"
find . -type f -size +10M > .dockerignore
docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t final .
FINAL_ID=$(docker inspect final:latest --format='{{index .Id}}')

echo "Final image: $FINAL_ID"
cog push