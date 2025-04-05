import os
import json
from PIL import Image

def crop_and_save_bounding_boxes(dataset_path, output_path="cropped_data", 
                                 num_folders=None, num_files_per_folder=None):

    os.makedirs(output_path, exist_ok=True)

    all_items = os.listdir(dataset_path)
    folders = [f for f in sorted(all_items) if os.path.isdir(os.path.join(dataset_path, f))]

    # Limit folders if specified
    if num_folders is not None:
        folders = folders[:num_folders]

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        files = sorted(os.listdir(folder_path))
        bb_files = [f for f in files if f.endswith('_bb.png')]

        # Limit number of files if specified
        if num_files_per_folder is not None:
            bb_files = bb_files[:num_files_per_folder]

        for bb_file in bb_files:
            base_name = bb_file.replace('_bb.png', '')
            txt_file = f"{base_name}.txt"
            txt_path = os.path.join(folder_path, txt_file)
            image_path = os.path.join(folder_path, bb_file)

            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    data = json.load(f)

                bbox = data['label'][0]
                img = Image.open(image_path)
                width, height = img.size

                # Convert normalized coords to absolute
                left = int(bbox['topX'] * width) + 2
                top = int(bbox['topY'] * height) + 2
                right = int(bbox['bottomX'] * width) - 1
                bottom = int(bbox['bottomY'] * height) - 1
                cropped_img = img.crop((left, top, right, bottom))

                # Save directly to the output_path with folder name prefix to avoid name clashes
                save_name = f"{base_name}_cropped.png"
                save_path = os.path.join(output_path, save_name)
                cropped_img.save(save_path)
