import os
from collections import defaultdict

OUTPUT_PATH = 'out_data/'

def check_dataset_balance(output_path=OUTPUT_PATH):
    """
    Checks whether each category (identified by the filename prefix, assumed to be the PLU number)
    in the output folder has the same number of images as the category with the highest count.
    
    Returns:
        missing_dict (dict): A dictionary where keys are PLU numbers and values are the number 
                             of images missing to reach the maximum count.
        is_balanced (bool): True if every category has the maximum count, False otherwise.
    """
    # List all PNG files in the output folder
    all_files = [f for f in os.listdir(output_path) if f.endswith('.png')]
    
    # Count images per category (assumes category is the first part of the filename before an underscore)
    category_counts = defaultdict(int)
    for filename in all_files:
        parts = filename.split('_')
        if parts:
            category = parts[0]
            category_counts[category] += 1
    
    if not category_counts:
        print("No images found in the output folder.")
        return {}, True

    # Find the maximum image count among categories
    max_count = max(category_counts.values())
    
    # Build a dictionary of missing images per category
    missing_dict = {}
    is_balanced = True
    for category, count in sorted(category_counts.items()):
        missing = max_count - count
        missing_dict[category] = missing
        if missing > 0:
            is_balanced = False
    
    return missing_dict, is_balanced

if __name__ == '__main__':
    missing_images, balanced = check_dataset_balance(OUTPUT_PATH)
    print("Missing images per category (PLU):")
    for plu, missing_count in missing_images.items():
        print(f"Category {plu}: Missing {missing_count} images")
    print("\nDataset balanced:", balanced)

