import argparse
from src.cropping import crop_and_save_bounding_boxes
from src.data_loader import load_files_to_dataframe
from src.predict import load_model_and_encoder, predict_images

output_path = "cropped_data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True, help="Path to the validation folder")
    args = parser.parse_args()

    val_dir = args.val_dir

    crop_and_save_bounding_boxes(val_dir, output_path=output_path)

    df = load_files_to_dataframe(output_path)


    # Fix: unpack the returned model and encoder
    model, encoder = load_model_and_encoder("src/efficientnet_best_model.h5", "src/label_encoder.pkl")
    predictions = predict_images(model, encoder, df)

    print(predictions[['label', 'predicted_label', 'confidence']])

    accuracy = (predictions['label'] == predictions['predicted_label']).mean()
    print(f"Accuracy: {accuracy:.2%}")

