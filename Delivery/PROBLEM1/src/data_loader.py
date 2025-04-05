import os
import pandas as pd
from PIL import Image

def get_label_from_filename(filename):
    """Extracts the label (PLU) from the filename (split by '-')."""
    return filename.split('-')[0]

def load_files_to_dataframe(path):
    """
    Loads image files from the given path into a DataFrame.
    Each row contains a PIL image and its corresponding label.
    """
    data = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.png'):
            label = get_label_from_filename(filename)
            image_path = os.path.join(path, filename)
            image = Image.open(image_path)
            data.append({'image': image, 'label': label})
    return pd.DataFrame(data)