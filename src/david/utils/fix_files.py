import glob
import os


def delete_single_line_json_files(directory):
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Check if the file is a JSON file
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            # Open the file and read the lines
            with open(file_path, "r") as file:
                lines = file.readlines()
                # Check if the file has only one line
                if (
                    (len(lines) == 0)
                    | (len(lines) == 1)
                    | (len(lines) == 2)
                    | (len(lines) == 3)
                ):
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted '{file_path}' as it contains only one line.")


directory = "/home/ddavilag/Desktop/mimic/data"
delete_single_line_json_files(directory)
print(len(glob.glob("/home/ddavilag/Desktop/mimic/data/*.json")))
print(len(glob.glob("/home/ddavilag/Desktop/mimic/data/*.txt")))
