#!/bin/bash

# Download and unzip.
curl -L https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip --output images_evaluation.zip
curl -L https://github.com/brendenlake/omniglot/raw/master/python/one-shot-classification/all_runs.zip --output all_runs.zip
unzip images_evaluation.zip
unzip all_runs.zip -d all_runs

# Convert to bitmap and match using md5 hash.
# This is necessary because the hashes of the PNGs do not match for some reason!
bash convert_bmp.sh images_evaluation
bash convert_bmp.sh all_runs
bash match.sh images_evaluation all_runs >run_images.txt

# Extract run and alphabet from filename.
cat run_images.txt | awk '{print $2,$3}' | \
    sed -e 's:all_runs/\([^/]*\)/[^ ]*\.bmp:\1:' | \
    sed -e 's:images_evaluation/\([^/]*\)/[^ ]*\.bmp:\1:' | \
    uniq >run_alphabets.txt
