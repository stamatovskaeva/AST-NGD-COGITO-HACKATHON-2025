#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import sys
import argparse
import pandas as pd



output_path = "cropped_data"

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess the image for the model.
    Args:
        img: Input image (numpy array).
        target_size: Target size for resizing the image.
    Returns:
        Preprocessed image (numpy array).
    """
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# function to chedck future frames for the same product
def check_future_frames(label, frame_number, frame_depth=10):
    # check average confidence for the label in future 10 frames
    future_frames = detections_by_frame[frame_number:frame_number + frame_depth]
    confidences = []
    for ff in future_frames:
        for detection in ff[0]:
            if detection['label'] == label:
                confidences.append(detection['conf'])
    return np.mean(confidences) if confidences else 0

def is_safe_period_coming(frame_number, frame_depth=10, threshold=0.3):
    """
    Check if the best average confidence for any label in the next frame_depth frames
    is below the given threshold. Used to determine if it's safe to re-enable detection.
    """
    future_frames = detections_by_frame[frame_number:frame_number + frame_depth]
    label_confidences = {}

    for ff in future_frames:
        for detection in ff[0]:
            label = detection['label']
            conf = detection['conf']
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(conf)

    if not label_confidences:
        return True  # no detections = safe

    best_avg_conf = max(
        np.mean(conf_list) for conf_list in label_confidences.values()
    )
    return best_avg_conf < threshold

receipt = []

plu_mapping = {
    '4011': 'Bananer Bama',
    '4015': 'Epler Røde',
    '4088': 'Paprika Rød',
    '4196': 'Appelsin',
    '94011': 'Bananer Økologisk',
    '90433917': 'Red Bull Regular 250ml boks',
    '90433924': 'Red Bull Sukkerfri 250ml boks',
    '7020097009819': 'Karbonadedeig 5% u/Salt og Vann 400g Meny',
    '7020097026113': 'Kjøttdeig Angus 14% 400g Meny',
    '7023026089401': 'Ruccula 65g Grønn&Frisk',
    '7035620058776': 'Rundstykker Grove Fullkorn m/Frø Rustikk 6stk 420g',
    '7037203626563': 'Leverpostei Ovnsbakt Orginal 190g Gilde',
    '7037206100022': 'Kokt Skinke Ekte 110g Gilde',
    '7038010009457': 'Yoghurt Skogsbær 4x150g Tine',
    '7038010013966': 'Norvegia 26% skivet 150g Tine',
    '7038010021145': 'Jarlsberg 27% skivet 120g Tine',
    '7038010054488': 'Cottage Cheese Mager 2% 400g Tine',
    '7038010068980': 'Yt Protein Yoghurt Vanilje 430g Tine',
    '7039610000318': 'Frokostegg Frittgående L 12stk Prior',
    '7040513000022': 'Gulrot 750g Beger',
    '7040513001753': 'Gulrot 1kg pose First Price',
    '7040913336684': 'Evergood Classic Filtermalt 250g',
    '7044610874661': 'Pepsi Max 0,5l flaske',
    '7048840205868': 'Frokostyoghurt Skogsbær 125g pose Q',
    '7071688004713': 'Original Havsalt 190g Sørlandschips',
    '7622210410337': 'Kvikk Lunsj 3x47g Freia'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="path to the video file")
    args = parser.parse_args()

    video_path = args.video

    model_path = 'src/efficientnet_best_model.h5'
    encoder_path = 'src/label_encoder.pkl'

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    if not os.path.exists(encoder_path):
        print(f"Error: Label encoder not found at {encoder_path}")
        sys.exit(1)

    model = load_model(model_path)
    label_encoder = joblib.load(encoder_path)

    num_classes = len(label_encoder.classes_)
    print(f"Loaded model with {num_classes} classes: {label_encoder.classes_}")

    video_path = video_path
    output_receipt_file = 'generated_receipt.csv'

    area_threshold = 2000
    distance_threshold = 60
    disappearance_threshold = 1.0
    confidence_threshold = 0.9

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read the video.")
        cap.release()
        sys.exit(1)

    frame_height, frame_width = first_frame.shape[:2]
    print(f"Frame dimensions: {frame_width} x {frame_height}")
    print(f"Video FPS: {fps}")

    roi_y = int(frame_height * 0.5)
    roi_h = int(frame_height * 0.5)
    roi_x = int(frame_width * 0.333)
    roi_w = int(frame_width * 0.333)

    print(f"Using ROI -> X:{roi_x}, Y:{roi_y}, Width:{roi_w}, Height:{roi_h}")

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    active_detections = []
    detections_by_frame = []
    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_time = frame_number / fps
        roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        fgmask = fgbg.apply(roi_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections_in_frame = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_threshold:
                x, y, w, h = cv2.boundingRect(cnt)
                crop = roi_frame[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                processed_crop = preprocess_image(crop)
                preds = model.predict(processed_crop, verbose=0)
                conf = np.max(preds)
                if conf < confidence_threshold:
                    continue
                pred_class = np.argmax(preds, axis=1)[0]
                label = label_encoder.classes_[pred_class]
                centroid = (x + w // 2, y + h // 2)
                detections_in_frame.append({'label': label, 'centroid': centroid, 'bbox': (x, y, w, h), 'conf': conf})

                real_x = x + roi_x
                real_y = y + roi_y
                cv2.rectangle(frame, (real_x, real_y), (real_x+w, real_y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (real_x, real_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        detections_by_frame.append(([{'label': det['label'], 'conf': det['conf']} for det in detections_in_frame], current_time))

        for det in detections_in_frame:
            matched = False
            for active in active_detections:
                if det['label'] == active['label']:
                    distance = euclidean_distance(det['centroid'], active['centroid'])
                    if distance < distance_threshold:
                        active['centroid'] = det['centroid']
                        active['last_seen_time'] = current_time
                        matched = True
                        break
            if not matched:
                active_detections.append({
                    'label': det['label'],
                    'centroid': det['centroid'],
                    'last_seen_time': current_time,
                    'bbox': det['bbox']
                })

        for active in active_detections.copy():
            if current_time - active['last_seen_time'] > disappearance_threshold:
                product_label = active['label']
                active_detections.remove(active)

        cv2.imshow('Full Frame with Detections', frame)
        cv2.imshow('ROI Frame', roi_frame)
        cv2.imshow('Foreground Mask', fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

    # Finalize lingering detections
    for active in active_detections:
        product_label = active['label']

    frame_number = 0
    enccountered_safe_period = True

    while frame_number < len(detections_by_frame):

        if not enccountered_safe_period:
            if is_safe_period_coming(frame_number, 30):
                enccountered_safe_period = True
                frame_number += 30
            else :
                frame_number += 5
            continue

        dbf = detections_by_frame[frame_number]
        current_detections = dbf[0]
        time = dbf[1]

        confidences = []
        for detection in current_detections:
            label = detection['label']
            future_conf = check_future_frames(label, frame_number, 20)
            confidences.append((label, future_conf))
        
        if confidences:
            max_label = max(confidences, key=lambda x: x[1])
            if max_label[1] > 0.98:
                receipt.append((plu_mapping[max_label[0]], time))
                frame_number += 10
                enccountered_safe_period = False
                continue
        frame_number += 1   

    print("Receipt:")
    for item in receipt:
        print(f"{item[0]}, {item[1]:.2f} seconds")

        

    # Format Time column to two decimals
    receipt_df = pd.DataFrame(receipt, columns=['Product', 'Time'])
    receipt_df['Time'] = receipt_df['Time'].map(lambda x: f"{x:.2f}")

    # Save to CSV
    receipt_df.to_csv(output_receipt_file, index=False)
    print(f"Receipt saved to {output_receipt_file}")