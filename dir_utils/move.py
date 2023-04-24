"""Move files with keywords in their names to a new directory."""

from __future__ import annotations
from pathlib import Path
import os
import shutil

def move_files_with_keywords(src_dir, rs_dest_dir, sf_dest_dir):
    # Ensure destination directories exist
    os.makedirs(rs_dest_dir, exist_ok=True)
    os.makedirs(sf_dest_dir, exist_ok=True)

    # Iterate over files in the source directory
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)

        # Check if the file contains "RS" or "SF" and move accordingly
        if os.path.isfile(file_path):
            if 'RS' in filename:
                shutil.move(file_path, os.path.join(rs_dest_dir, filename))
                print(f"Moved {filename} to {rs_dest_dir}")
            elif '_SF' in filename:
                shutil.move(file_path, os.path.join(sf_dest_dir, filename))
                print(f"Moved {filename} to {rs_dest_dir}")

if __name__ == "__main__":

    main_dir = Path(r"C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\allfiles")
    rs_dest_dir = Path(r"C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\org\nex_rs")
    sf_dest_dir = Path(r"C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\org\nex_sf")
    move_files_with_keywords(main_dir, rs_dest_dir, sf_dest_dir)