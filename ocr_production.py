import os
import argparse
import cv2
import numpy as np
import csv
from PIL import ImageFont, ImageDraw, Image
from tensorflow import keras
from keras.preprocessing.image import load_img
import arabic_reshaper
from bidi.algorithm import get_display
from ultralytics import YOLO
import time
import pandas as pd

# Load YOLO models
ocr_model = YOLO(r'D:\GP\runs\detect\yolo11m_car_plate\weights\best.pt')
plate_detector_model = YOLO(r'D:\GP\runs\detect\yolo_car_plate\weights\best.pt')

# Define font path
font_path = r"D:\GP\alfont_com_arial-1.ttf"

# Define class labels mapping
CLASS_LABELS_MAPPING = {
    0: "٠", 1: "١", 2: "٢", 3: "٣", 4: "٤", 5: "٥", 6: "٦", 7: "٧",
    8: "ح", 9: "٨", 10: "٩", 11: "ط", 12: "ظ", 13: "ع", 14: "أ", 15: "ب",
    16: "ض", 17: "د", 18: "ف", 19: "غ", 20: "ه", 21: "ج", 22: "ك", 23: "خ",
    24: "ل", 25: "م", 26: "ن", 27: "ق", 28: "ر", 29: "ص", 30: "س", 31: "ش",
    32: "ت", 33: "ث", 34: "و", 35: "ي", 36: "ذ", 37: "ز"
}

# Define colors for visualization
COLORS = [
    (255, 0, 0), (34, 75, 12), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (21, 52, 72), (66, 50, 168)
]

def draw_arabic_text(image, text, position, color):
    """
    Draw Arabic text on an image with background rectangle
    """
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, 40)
    bbox = draw.textbbox(position, bidi_text, font=font)
    draw.rectangle(bbox, fill=color)
    draw.text(position, bidi_text, font=font, fill="black")
    return np.array(img_pil)

def reverse_arabic(text):
    """
    Reverse Arabic text while preserving numbers and adding spaces between characters
    """
    segments = []
    current_segment = ""
    is_number = text[0].isdigit()

    for char in text:
        if char.isdigit() == is_number:
            current_segment += char
        else:
            if current_segment:
                segments.append(current_segment)
            current_segment = char
            is_number = not is_number

    segments.append(current_segment)

    reversed_text = ""
    for i, segment in enumerate(segments):
        if segment[0].isdigit():
            # Add space before and after numbers
            if i > 0:  # Add space before number if not first segment
                reversed_text += " "
            reversed_text += segment
            if i < len(segments) - 1:  # Add space after number if not last segment
                reversed_text += " "
        else:
            # Add space between each Arabic character
            reversed_text = " ".join(segment[::-1]) + reversed_text

    return reversed_text.strip()  # Remove any extra spaces at the beginning or end

def detect_roi(image_path, plate_detector_model):
    try:
        # Perform prediction
        results = plate_detector_model(image_path)

        if len(results) == 0:
            raise ValueError("No boxes predicted for image.")

        # Extract box coordinates from results
        box = results[0].boxes.data[0].tolist()
        x1, y1, x2, y2 = box[0:4]
        x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]

        # Load original image
        img = np.array(load_img(image_path))

        # Extract region of interest (ROI)
        roi = img[y1:y2, x1:x2]
        roi = cv2.resize(roi, (220, 220))

        # Convert ROI to BGR format
        image_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

        # Save ROI image in "plates" folder
        roi_filename = os.path.basename(image_path)
        roi_filename = os.path.splitext(roi_filename)[0]
        roi_path = os.path.join("./plates", f"{roi_filename}.jpg")
        cv2.imwrite(roi_path, image_bgr)

        return roi_path, image_bgr, (x1, y1, x2, y2)
    except Exception as e:
        print(f"Failed to predict boxes for image: {image_path}")
        print(f"Error: {str(e)}")
        return None, None, None

def draw_char(roi_path, roi, ocr_model):
    if roi_path is None:
        print("No plate detected.")
        return None, False, ''

    try:
        # Perform prediction
        results = ocr_model(roi_path)

        detected_plate = False
        predicted_label = ''

        # Process results list
        for i, result in enumerate(results):
            # Sort boxes based on their x-coordinate (left to right)
            sorted_boxes = sorted(result.boxes.data, key=lambda box: box[0])

            for j, box in enumerate(sorted_boxes):
                # Extract box coordinates, confidence, and class
                x1, y1, x2, y2, conf, cls = box[:6]

                # Convert class number to Arabic character
                class_label = CLASS_LABELS_MAPPING.get(int(cls.item()), 'Unknown')

                # Determine the color for the character
                color_index = j % len(COLORS)
                color = COLORS[color_index]

                # Draw rectangle around the character
                roi = cv2.rectangle(roi, (int(x1), int(y1)),
                                    (int(x2), int(y2)), color, 2)

                # Add class label text with background
                roi = draw_arabic_text(
                    roi, class_label, (int(x1), int(y1 - 40)), color)

                predicted_label += class_label

                detected_plate = True

        # Reverse the Arabic text for correct display
        predicted_label = reverse_arabic(predicted_label)
        print(predicted_label)
        return roi, detected_plate, predicted_label
    except Exception as e:
        print(f"Error in drawing characters on ROI: {str(e)}")
        return None, False, ''

def process_video(video_path, model):
    """
    Process a video file for license plate detection
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    size = (frame_width, frame_height)

    processed_video_path = os.path.join(os.getcwd(), 'processed_video.mp4')
    out = cv2.VideoWriter(processed_video_path, fourcc, original_fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for i, result in enumerate(results):
            sorted_boxes = sorted(result.boxes.data, key=lambda box: box[0])

            for j, box in enumerate(sorted_boxes):
                x1, y1, x2, y2, conf, cls = box[:6]
                class_label = CLASS_LABELS_MAPPING.get(int(cls), 'Unknown')
                color_index = j % len(COLORS)
                color = COLORS[color_index]
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                frame = draw_arabic_text(
                    frame, class_label, (int(x1), int(y1 - 40)), color)

        out.write(frame)
        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return processed_video_path

def process_all_images(input_folder, output_folder, plate_detector_model, ocr_model):
    """
    Process all images in a folder and save detailed results to CSV
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create plates folder if it doesn't exist
    plates_folder = os.path.join(output_folder, "plates")
    if not os.path.exists(plates_folder):
        os.makedirs(plates_folder)

    # Open CSV file for writing with UTF-8 encoding
    csv_path = os.path.join(output_folder, 'detailed_results.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header with detailed information
        csv_writer.writerow([
            'Image_Name',
            'Image_Path',
            'Plate_Detected',
            'Detected_Text',
            'x1', 'y1', 'x2', 'y2',
            'Processed_Image_Path',
            'Processing_Time',
            'Confidence_Score'
        ])

        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])

        total_images = len(image_files)
        print(f"Found {total_images} images to process")

        # Process each image
        for idx, filename in enumerate(image_files, 1):
            start_time = time.time()
            image_path = os.path.join(input_folder, filename)
            print(f"Processing image {idx}/{total_images}: {filename}")

            try:
                # Detect ROI
                roi_path, roi, coordinates = detect_roi(image_path, plate_detector_model)
                
                if roi is not None:
                    # Draw characters on ROI
                    roi_with_chars, plate_detected, predicted_label = draw_char(roi_path, roi, ocr_model)
                    
                    # Save processed image
                    if roi_with_chars is not None:
                        output_path_roi = os.path.join(plates_folder, f"plate_{filename}")
                        cv2.imwrite(output_path_roi, roi_with_chars)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Get confidence score (you can modify this based on your model's output)
                    confidence_score = 1.0 if plate_detected else 0.0
                    
                    # Write to CSV
                    csv_writer.writerow([
                        filename,
                        image_path,
                        plate_detected,
                        predicted_label,
                        coordinates[0] if coordinates else 'N/A',
                        coordinates[1] if coordinates else 'N/A',
                        coordinates[2] if coordinates else 'N/A',
                        coordinates[3] if coordinates else 'N/A',
                        output_path_roi if roi_with_chars is not None else 'N/A',
                        f"{processing_time:.2f}",
                        confidence_score
                    ])
                else:
                    # Write information for undetected plates
                    csv_writer.writerow([
                        filename,
                        image_path,
                        False,
                        'N/A',
                        'N/A', 'N/A', 'N/A', 'N/A',
                        'N/A',
                        f"{time.time() - start_time:.2f}",
                        0.0
                    ])

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                # Write error information
                csv_writer.writerow([
                    filename,
                    image_path,
                    False,
                    f"Error: {str(e)}",
                    'N/A', 'N/A', 'N/A', 'N/A',
                    'N/A',
                    f"{time.time() - start_time:.2f}",
                    0.0
                ])

            # Add a small delay to prevent overloading
            time.sleep(0.1)

    print(f"Processing complete. Results saved to {csv_path}")
    return csv_path

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process all images and create detailed results CSV")
    parser.add_argument("-p", "--path", type=str,
                        help="Path to the folder of test images", 
                        default=r"D:\GP\EALPR Vechicles dataset\Vehicles")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to the output folder",
                        default="results")
    args = parser.parse_args()

    # Process all images and save results
    csv_path = process_all_images(args.path, args.output, plate_detector_model, ocr_model)
    
    # Display results summary
    df = pd.read_csv(csv_path)
    print("\nResults Summary:")
    print(f"Total images processed: {len(df)}")
    print(f"Plates detected: {df['Plate_Detected'].sum()}")
    print(f"Detection rate: {(df['Plate_Detected'].sum() / len(df) * 100):.2f}%")
    print("\nFirst few results:")
    print(df.head())
