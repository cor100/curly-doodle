URL="https://dcapswoz.ict.usc.edu/wwwedaic/"
SAVE_DIR="/Users/cynthia/curly-doodle"
PYTHON_SCRIPT="/Users/cynthia/curly-doodle/scripts/load_duckdb.py"

echo "Fetching file list from $URL..."

wget -r -np -nH --cut-dirs=1 -P "$SAVE_DIR" "$URL"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to download files. Please check the URL or your internet connection."
    exit 1
fi
log "Checking for compressed files to extract..."
cd "$SAVE_DIR"
for file in *.zip *.tar.gz; do
    if [ -e "$file" ]; then
        log "Extracting $file..."
        if [[ "$file" == *.zip ]]; then
            unzip -o "$file"
        elif [[ "$file" == *.tar.gz ]]; then
            tar -xvzf "$file"
        fi
        log "Loading extracted data from $file into DuckDB."
        python3 "$PYTHON_SCRIPT" "$SAVE_DIR/$file"
        if [ $? -ne 0 ]; then
            error "Failed to process $file into DuckDB."
        fi
        rm -f "$file"
        log "$file extracted and removed."
    fi
done

cd ..

echo "Download completed"

