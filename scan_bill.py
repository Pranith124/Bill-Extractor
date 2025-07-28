# scan_bill.py
import cv2
import pandas as pd
from ultralytics import YOLO
import easyocr
import os

# --- CONFIGURATION ---
# Path to your trained YOLOv8 model
MODEL_PATH = 'runs/detect/train/weights/best.pt'
# Name of the output Excel file
EXCEL_FILE = 'bills_data.xlsx'
# Initialize the OCR reader
reader = easyocr.Reader(['en']) # 'en' for English

# --- LOAD THE TRAINED MODEL ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure the 'best.pt' file is in the correct directory.")
    exit()

def save_to_excel(data_dict):
    """Saves the extracted data dictionary to an Excel file."""
    print(f"Saving data: {data_dict}")
    
    # Convert dictionary to a pandas DataFrame
    new_data_df = pd.DataFrame([data_dict])
    
    # If the Excel file already exists, load it and append the new data
    if os.path.exists(EXCEL_FILE):
        existing_df = pd.read_excel(EXCEL_FILE)
        updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    else:
        # If the file doesn't exist, the new data is our dataframe
        updated_df = new_data_df
        
    # Save the updated DataFrame to Excel
    updated_df.to_excel(EXCEL_FILE, index=False)
    print(f"Data successfully saved to {EXCEL_FILE}")

def process_frame(frame):
    """Processes a single frame to detect, OCR, and save bill data."""
    # 1. DETECT objects using your custom YOLOv8 model
    results = model(frame)[0]
    
    # Get class names from the model
    class_names = results.names
    
    # Dictionary to hold extracted data
    extracted_data = {}

    # 2. OCR detected regions
    for res in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = res
        
        if score > 0.5: # Confidence threshold
            # Crop the detected region from the frame
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Use OCR to read text from the cropped image
            ocr_result = reader.readtext(cropped_image, detail=0, paragraph=False)
            
            if ocr_result:
                # Store the first line of OCR text for the corresponding class
                class_name = class_names[int(class_id)]
                extracted_data[class_name] = ocr_result[0]

    # 3. SAVE TO EXCEL if data was found
    if extracted_data:
        save_to_excel(extracted_data)
        return True
    else:
        print("No data found or confidence too low.")
        return False


# --- MAIN CAMERA LOOP ---
cap = cv2.VideoCapture(0) # 0 is the default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened. Press 's' to scan a bill. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live camera feed
    cv2.imshow('Bill Scanner - Press "s" to Scan, "q" to Quit', frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'): # 's' key to scan
        print("\nScanning frame...")
        if process_frame(frame):
            print("Scan successful! Ready for next bill.")
        else:
            print("Scan failed. Please try again.")
            
    elif key == ord('q'): # 'q' key to quit
        break

cap.release()
cv2.destroyAllWindows()
print("Program terminated.")