# Grocery Product Recognition

This repository contains a solution for classifying grocery products into their respective PLU/GTIN codes using an EfficientNet-based model. The code handles:

- **Data Splitting**: Splits a set of cropped images into training and testing folders.
- **Cropping**: Uses bounding box annotations to crop product images from larger images.
- **Model Training**: Trains and evaluates an EfficientNetB0-based model.
- **Inference**: Runs inference on a new validation directory.

## Project Structure
DELIVERY
└── PROBLEM1
    └── README.md
    └── main.py
    └── requirements.txt
    └── model/
       └── efficientnet_best_model.h5   # saved model (after training)
    └── src/
        └── cropping.py
        └── data_loader.py
        └── model_evaluation.py


## Setup Instructions

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt

2. Prepare Your Data
	•	Place your original images (with bounding box .txt files) in a folder named data/.
	•	Run the cropping code to produce cropped images into cropped_data/.

3. Run the Solution

```py

python main.py

```
By default, this will:
	•	Crop images (if desired)
	•	Evaluate the model on the test set

4.	Inference on a New Validation Folder

```py

python main.py --val_dir /path/to/validation_folder

```
This will load the saved model and produce predictions for each image in the provided folder.

## Model Details
	•	Architecture: EfficientNetB0 (ImageNet pretrained, with frozen layers).
	•	Classifier Head: Global Average Pooling -> Dense(128, ReLU) -> Dense(num_classes, Softmax).
	•	Loss: Sparse Categorical Crossentropy.
	•	Optimizer: Adam with learning rate 0.001.
	•	Callbacks: EarlyStopping and ModelCheckpoint.

## Known Limitations / Future Work
	•	Fine-tuning the base EfficientNet layers could yield higher accuracy.
	•	Additional augmentation techniques (color jitter, random crops, etc.) may help.
	•	Saving and reloading the label encoder is recommended for more robust inference.

Enjoy using this solution!