#!/bin/bash

# 1. Setup Directories
export SEMIDENSE_SAMPLE_PATH="/tmp/semidense_samples"
mkdir -p "$SEMIDENSE_SAMPLE_PATH/ase"
mkdir -p "$SEMIDENSE_SAMPLE_PATH/aea"

# 2. Base URLs
export ASE_BASE_URL="https://www.projectaria.com/async/sample/download/?bucket=ase&filename="
export AEA_BASE_URL="https://www.projectaria.com/async/sample/download/?bucket=aea&filename="

# 3. Fixed Options (Removed -O, kept -L for redirects)
# -L: Follow redirects (Critical for these URLs)
# -# : Show progress bar
export OPTIONS="-L -#"

echo "Downloading ASE examples..."
curl $OPTIONS -o "$SEMIDENSE_SAMPLE_PATH/ase/ase_examples.zip" "${ASE_BASE_URL}ase_examples.zip"

echo "Downloading AEA (Real-world) examples..."
curl $OPTIONS -o "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script1_seq1_rec1.zip" "${AEA_BASE_URL}loc1_script1_seq1_rec1.zip"
curl $OPTIONS -o "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec1_10s_sample.zip" "${AEA_BASE_URL}loc1_script2_seq1_rec1_10s_sample.zip"
curl $OPTIONS -o "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec2_10s_sample.zip" "${AEA_BASE_URL}loc1_script2_seq1_rec2_10s_sample.zip"

echo "Unzipping..."
# ASE
unzip -q -o "$SEMIDENSE_SAMPLE_PATH/ase/ase_examples.zip" -d "$SEMIDENSE_SAMPLE_PATH/ase/ase_examples"
# AEA
unzip -q -o "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script1_seq1_rec1.zip" -d "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script1_seq1_rec1"
unzip -q -o "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec1_10s_sample.zip" -d "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec1_10s_sample"
unzip -q -o "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec2_10s_sample.zip" -d "$SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec2_10s_sample"

echo "✅ Download and extraction complete at $SEMIDENSE_SAMPLE_PATH"