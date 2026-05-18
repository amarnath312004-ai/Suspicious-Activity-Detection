import os
import cv2
from ultralytics import YOLO
import pandas as pd

# Load your YOLO model
model = YOLO("yolo11s-pose.pt")

# Video path
cap = cv2.VideoCapture(r'D:\SUSPICIOUS ACTIVITY DETECTION\yolo11_suspicious_activity-main\susp1.mp4')

# Get video properties
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
seconds = round(frames / fps)

frame_total = 2000
i = 0
a = 1543  # Start from 1543

all_data = []

# Define output path for cropped images
output_path_dir = r'D:\SUSPICIOUS ACTIVITY DETECTION\yolo11_suspicious_activity-main\images 1'
os.makedirs(output_path_dir, exist_ok=True)  # Ensure directory exists

# Directory for full-frame images
full_frame_dir = r'D:\SUSPICIOUS ACTIVITY DETECTION\yolo11_suspicious_activity-main\images'
os.makedirs(full_frame_dir, exist_ok=True)  # Ensure directory exists

while cap.isOpened():
    # Set the position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds / frame_total) * 1000)))
    flag, frame = cap.read()

    if not flag:
        break

    # Save full frame image
    image_path = os.path.join(full_frame_dir, f'img_{i}.jpg')
    cv2.imwrite(image_path, frame)

    # Run YOLO detection
    results = model(frame, verbose=False)

    for r in results:
        bound_box = r.boxes.xyxy  # Get bounding boxes
        conf = r.boxes.conf.tolist()  # Confidence score
        keypoints = r.keypoints.xyn.tolist()  # Human keypoints

        for index, box in enumerate(bound_box):
            if conf[index] > 0.75:
                x1, y1, x2, y2 = map(int, box.tolist())
                cropped_person = frame[y1:y2, x1:x2]  # Crop person
                output_path = os.path.join(output_path_dir, f'person_nn_{a}.jpg')

                # Prepare keypoint data
                data = {'image_name': f'person_nn_{a}.jpg'}
                for j, point in enumerate(keypoints[index]):
                    data[f'x{j}'] = point[0]
                    data[f'y{j}'] = point[1]

                all_data.append(data)
                cv2.imwrite(output_path, cropped_person)  # Save cropped image
                a += 1  # Increment image number for the next person

    i += 1

print(f"Total frames processed: {i - 1}, Total cropped images saved: {a - 1}")
cap.release()
cv2.destroyAllWindows()

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Path to your CSV file
csv_file_path = r'D:\SUSPICIOUS ACTIVITY DETECTION\yolo11_suspicious_activity-main\keypoint.csv'

# Check if the file exists to determine whether to append or create new
if not os.path.isfile(csv_file_path):
    df.to_csv(csv_file_path, index=False)  # Create new file if it doesn't exist
else:
    df.to_csv(csv_file_path, mode='a', header=False, index=False)  # Append if it exists

print(f"Keypoint data saved to {csv_file_path}")
