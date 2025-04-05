import os
import pandas as pd
from PIL import Image

# IMPROT 

def get_label_from_filename(filename):
    return filename.split('-')[0]

def load_files_to_dataframe(path):
    data = []
    for filename in os.listdir(path):
        label = get_label_from_filename(filename)
        image_path = os.path.join(path, filename)
        with Image.open(image_path) as img:
            img_copy = img.copy()  # Copy the image before closing the file
        data.append({'image': img_copy, 'label': label})
    return pd.DataFrame(data)