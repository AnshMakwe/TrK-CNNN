import zipfile
import os
zip_path = "train_and_validation_sets.zip"  # Change to your file's name
extract_path = "."  # Directory where files will be extracted

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Dataset extracted to {extract_path}")