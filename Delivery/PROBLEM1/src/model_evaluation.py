import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_files_to_dataframe

# Define global paths
MODEL_PATH = os.path.join("model", "efficientnet_best_model.h5")
TEST_DIR = "cropped_testing"

def preprocess_images(images, target_size=(224, 224)):
    processed_images = []
    for img in images:
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        processed_images.append(img_array)
    return np.array(processed_images)

def run_evaluation():
    """
    Loads the pretrained model and evaluates it on the test set located in TEST_DIR.
    """
    if not os.path.exists(MODEL_PATH):
        print("No trained model found at", MODEL_PATH)
        return

    model = load_model(MODEL_PATH)
    test_df = load_files_to_dataframe(TEST_DIR)
    X_test = test_df['image']
    y_test = test_df['label']

    # Preprocess images
    X_test_processed = preprocess_images(X_test)

    # Encode labels (Note: In a robust solution, save and reload the encoder)
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_encoded, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Generate predictions and evaluation metrics
    predictions = model.predict(X_test_processed)
    y_pred_classes = np.argmax(predictions, axis=1)

    print("Classification Report:")
    print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test_encoded, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def run_inference(val_dir):
    """
    Loads the pretrained model and runs inference on images in the provided folder.
    """
    if not os.path.exists(MODEL_PATH):
        print("No trained model found at", MODEL_PATH)
        return

    model = load_model(MODEL_PATH)
    val_df = load_files_to_dataframe(val_dir)
    X_val = val_df['image']

    X_val_processed = preprocess_images(X_val)
    predictions = model.predict(X_val_processed)
    y_pred_classes = np.argmax(predictions, axis=1)

    print("Inference results:")
    for i, (idx, row) in enumerate(val_df.iterrows()):
        print(f"Image {i}: Predicted class index {y_pred_classes[i]}")