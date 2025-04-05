import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib  # use joblib instead of pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# Load the saved model and label encoder
def load_model_and_encoder(model_path, encoder_path):
    model = load_model(model_path)
    
    label_encoder = joblib.load(encoder_path)

    # Optional: check type if needed
    # from sklearn.preprocessing import LabelEncoder
    # if not isinstance(label_encoder, LabelEncoder):
    #     raise TypeError("Loaded object is not a LabelEncoder")

    return model, label_encoder


# Preprocess images for EfficientNet
def preprocess_images(images, target_size=(224, 224)):
    processed_images = []

    for img in images:
        if not isinstance(img, Image.Image):
            raise ValueError("All entries in df['image'] must be PIL.Image.Image instances")

        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        processed_images.append(img_array)

    return np.array(processed_images)


# Make predictions using the loaded model
def predict_images(model, label_encoder, df):
    if 'image' not in df.columns:
        raise KeyError("DataFrame must contain an 'image' column with PIL.Image.Image objects.")

    images = df['image'].tolist()
    processed_images = preprocess_images(images)

    predictions = model.predict(processed_images, verbose=0)

    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)

    df['predicted_label'] = predicted_labels
    df['confidence'] = np.max(predictions, axis=1)

    return df


# Example usage (optional testing block)
if __name__ == "__main__":
    # Example paths
    model_path = "src/efficientnet_best_model.h5"
    encoder_path = "src/label_encoder.pkl"

    # Load model and encoder
    model, label_encoder = load_model_and_encoder(model_path, encoder_path)

    # Example: Load some images
    from PIL import Image
    test_image_paths = ["example1.jpg", "example2.jpg"]  # update with your image paths
    images = [Image.open(p) for p in test_image_paths]

    # Create DataFrame
    df = pd.DataFrame({'image': images})

    # Predict
    df_with_preds = predict_images(model, label_encoder, df)
    print(df_with_preds[['predicted_label', 'confidence']])
