import os
import re

def delete_substrings(filename, substr: str):
    # Remove substring from filename
    modified_filename = filename.replace(substr, "")

    # Remove the second occurrence of "2018"
    count_2018 = modified_filename.count("2018")
    if count_2018 > 1:
        first_index = modified_filename.find("2018")
        second_index = modified_filename.find("2018", first_index + 1)
        modified_filename = modified_filename[:second_index] + modified_filename[second_index + 4:]

    return modified_filename

def rename_files(filename):
    def repl(match):
        prefix, number, separator, year, month, day, rest = match.groups()
        new_number = f"{int(number):02d}"
        new_separator = "_"
        formatted_year = f"20{year}" if len(year) == 2 else year
        formatted_date = f"{formatted_year}-{month}-{day}"
        return f"{prefix}{new_number}{new_separator}{formatted_date}_{rest}"

    # Apply the renaming function only to the matched part
    new_filename = re.sub(r'([A-Za-z]{3})(\d{1,2})([ _])(\d{2,4})(?:-|_)(\d{2})(?:-|_)?(\d{2})[ _]?(.*)', repl, filename)
    return new_filename

def delete_substrings_in_files_in_directory(directory, substr):
    for root, _, files in os.walk(directory):
        for file in files:
            old_file_path = os.path.join(root, file)
            new_filename = delete_substrings(file, substr)
            new_file_path = os.path.join(root, new_filename)

            # Rename the file only if the new and old filenames are different
            if old_file_path != new_file_path:
                try:
                    os.rename(old_file_path, new_file_path)
                except FileExistsError:
                    print(f"File already exists: {new_file_path}")
                print(f"Renamed: {old_file_path} -> {new_file_path}")

def rename_files_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            old_file_path = os.path.join(root, file)
            new_filename = rename_files(file)
            new_file_path = os.path.join(root, new_filename)

            # Rename the file only if the new and old filenames are different
            if old_file_path != new_file_path:
                try:
                    os.rename(old_file_path, new_file_path)
                except FileExistsError:
                    print(f"File already exists: {new_file_path}")
                print(f"Renamed: {old_file_path} -> {new_file_path}")

if __name__ == "__main__":

    directory_path = r"C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf"
    # rename_files_in_directory(directory_path)
    delete_substrings_in_files_in_directory(directory_path, "_OBEX")

