import duckdb
import sys
from io import BytesIO
import tarfile
import pandas as pd
import re

db_path = "/Users/cynthia/curly-doodle/edaic.duckdb"
# Validate arguments
if len(sys.argv) < 3:
    print("Usage: python load_duckdb.py --[tar/csv] <file_url>", file=sys.stderr)
    sys.exit(1)

file_type = sys.argv[1]
file_url = sys.argv[2]

conn = duckdb.connect(db_path)

def rename(path):
    base_name = path.split("/")[-1]  # Get the file name
    table_name = re.sub(r'\W|^(?=\d)', '_', base_name)  # Replace invalid characters and prefix digits
    return table_name.split('.')[0]  # Remove the file extension
    # return path.replace("https://","").replace("/","_").replace(".","_").replace("-","_")

def process_csv(stream, path):
    table_name = rename(path)
    df = pd.read_csv(stream)

    print(f"Loading {path} into DuckDB table {table_name}")
    # Drop table if it exists
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Insert the DataFrame into DuckDB
    conn.register("temp_table", df)
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_table")
    conn.unregister("temp_table")

if file_type == "--tar":
    # Handle .tar.gz files
    fileobj = BytesIO(sys.stdin.buffer.read())
    with tarfile.open(fileobj=fileobj, mode='r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.csv'):
                print(f"Processing {member.name} from {file_url}...")

                # Extract the file into memory
                extracted_file = tar.extractfile(member)
                if extracted_file is None:
                    print(f"Error extracting {member.name}, skipping...", file=sys.stderr)
                    continue
                # Use pandas to process the CSV content
                csv_stream = BytesIO(extracted_file.read())
                process_csv(csv_stream, f"{file_url}/{member.name}")

elif file_type == "--csv":
    # Handle standalone .csv files
    process_csv(sys.stdin.buffer, file_url)

print("Data loaded into DuckDB")
