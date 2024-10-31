from genericpath import isfile
import requests
import zipfile
from pydantic import HttpUrl
from pathlib import Path
import gdown

def validate_path_or_url(value):
    # Check if it's a valid URL
    try:
        HttpUrl(value)
        return value
    except Exception:
        pass

    # Check if it's a valid folder path
    path = Path(value)
    if path.is_dir() or path.is_file():
        return value
    raise ValueError("Input must be a valid URL or a folder path.")

def get_file_from(file_address, download_file_name=None):
    file_address = validate_path_or_url(file_address)
    
    if file_address.startswith('http'):
        # Determine destination path
        if download_file_name is None:
            raise ValueError("Please provide a download_file_name for remote files.")
        file_name, ext = download_file_name.rsplit('.', 1)
        local_path = f"/tmp/{file_name}.{ext}"

        # Use gdown for Google Drive links
        if "drive.google.com" in file_address:
            gdown.download(file_address, local_path, fuzzy=True)
        else:
            # Standard download for other URLs
            response = requests.get(file_address, stream=True)
            with open(local_path, "wb") as file:
                for chunk in response.iter_content(32768):  # Chunk size for large downloads
                    file.write(chunk)
        
        # Handle zip extraction if needed
        if ext == 'zip':
            file_dir = f"/tmp/{file_name}"
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(file_dir)
            local_path = file_dir
    else:
        local_path = file_address
    
    return local_path