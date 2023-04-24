import os
import shutil

def copy_files(source_folder, rs_dest_folder, sf_dest_folder):
    # Ensure the destination folders exist
    os.makedirs(rs_dest_folder, exist_ok=True)
    os.makedirs(sf_dest_folder, exist_ok=True)

    # Iterate through the files in the source folder
    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)

        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            # Copy the file to the respective folder based on the presence of "RS" or "SF" in the filename
            if "RS" in file_name:
                shutil.copy2(file_path, os.path.join(rs_dest_folder, file_name))
            elif "SF" in file_name:
                shutil.copy2(file_path, os.path.join(sf_dest_folder, file_name))

if __name__ == "__main__":

    source_folder = r"C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\allfiles"
    rs_dest_folder = r"C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\rs"
    sf_dest_folder = r"C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf"
    copy_files(source_folder, rs_dest_folder, sf_dest_folder)
