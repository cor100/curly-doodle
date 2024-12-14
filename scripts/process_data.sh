#!/bin/bash

URL="https://dcapswoz.ict.usc.edu/wwwedaic"
PYTHON_SCRIPT="/Users/cynthia/curly-doodle/scripts/load_duckdb.py"
SKIP=("labels2019.tar.gz")
log() {
    echo "[INFO] $1"
}

error() {
    echo "[ERROR] $1"
    exit 1
}

log "Fetching file list recursively from $URL..."

# Download the file list recursively (only the listing, no actual files)
wget --spider -r -np -nd "$URL" 2>&1 | grep -o "https://[^\"]*" > file_list.txt

if [ ! -s file_list.txt ]; then
    error "No files found at $URL or its subdirectories."
fi

log "Found files. Processing each file directly..."

# Loop through each .tar.gz file in the file list
while read -r file_url; do
    file_name=$(basename -- "$file_url")

    # Check if the file is in the skip list
    if [[ " ${SKIP[@]} " =~ " ${file_name} " ]]; then
        log "Skipping $file_name as it is in the skip list."
        continue
    fi

    log "Processing $file_url..."

    # Stream the file to the Python script
    if [[ "$file_url" == *.tar.gz ]]; then
        # Stream the tar.gz file to the Python script
        wget -qO - "$file_url" | python3 "$PYTHON_SCRIPT" --tar "$file_name"
    elif [[ "$file_url" == *.csv ]]; then
        # Process CSV files directly
        wget -qO - "$file_url" | python3 "$PYTHON_SCRIPT" --csv "$file_name"
    fi

    if [ $? -ne 0 ]; then
        # log "Python script failed for $file_url. Inspecting input..."
        # wget -qO - "$file_url" > debug_content.tar.gz
        error "Failed to process $file_url into DuckDB."
    fi

    log "Successfully processed $file_url."
done < file_list.txt

cd ..

log "All files processed successfully!"
