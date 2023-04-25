"""Move files with keywords in their names to a new directory."""

from __future__ import annotations
from pathlib import Path
import os
import openpyxl
import shutil

def move_files_with_keywords(src_dir, move_dir, search_list):
    # Ensure destination directories exist
    os.makedirs(move_dir, exist_ok=True)

    # Iterate over files in the source directory
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)

        # Check if the file contains any keywords in the search list and move accordingly
        if os.path.isfile(file_path):
            for keyword in search_list:
                if keyword in filename:
                    shutil.move(file_path, os.path.join(move_dir, filename))
                    print(f"Moved {filename} to {move_dir}")
                    search_list.remove(keyword)
                    break


if __name__ == "__main__":

    # Change these ------------------------
    # main_dir = Path(r"/path/to/search")
    # move_dir = Path(r"/your/path/to/move/to")
    workbook = Path(r"C:\Users\Flynn\OneDrive\Desktop\temp\workbook.xlsx")
    # --------------------------------------
    workbook = openpyxl.load_workbook(workbook)
    worksheet = workbook['FY 22 Billings']

    search_list = []
    for row in worksheet.iter_rows(min_row=2, min_col=3, values_only=True):
        value = str(row[0])
        search_list.append(value)

    # move_files_with_keywords(main_dir, move_dir, search_list)