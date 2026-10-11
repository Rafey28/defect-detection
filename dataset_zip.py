import shutil
import os

# Specify the directory path and output zip name
folder_to_zip = 'data/mvtec'
output_filename = 'data'

if os.path.exists(folder_to_zip):
    shutil.make_archive(output_filename, 'zip', folder_to_zip)
    print(f"Successfully created {output_filename}.zip! Ready for upload.")
else:
    print(f"Error: {folder_to_zip} does not exist. Check your path.")
    