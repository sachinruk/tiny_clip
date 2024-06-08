#!/usr/bin/env bash

# Array of URLs to be downloaded
URLS=(
    "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv"
    "https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv"
    "https://storage.googleapis.com/conceptual-captions-v1-1-labels/Image_Labels_Subset_Train_GCC-Labels-training.tsv"
)

# Array of output file names corresponding to the URLs
OUTPUT_FILES=(
    "GCC-training.tsv"
    "GCC-Validation.tsv"
    "GCC-Labels-training.tsv"
)

# Loop through each URL and download the file
for i in "${!URLS[@]}"; do
    URL=${URLS[$i]}
    OUTPUT_FILE=${OUTPUT_FILES[$i]}
    
    echo "Downloading $URL to $OUTPUT_FILE..."
    curl -L -o $OUTPUT_FILE $URL
    
    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download of $OUTPUT_FILE completed successfully."
    else
        echo "Download of $OUTPUT_FILE failed."
    fi
done

sed -i '1s/^/caption\turl\n/' ./GCC-training.tsv
img2dataset --url_list ./GCC-training.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256\
             --enable_wandb True
