import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone

# Define the path to the video file
video_path = r"D:\SUSPICIOUS-ACTIVITY-DETECTION-main\SUSPICIOUS-ACTIVITY-DETECTION-main\yolo11_suspicious_activity-main\susp1.mp4"


def detect_shoplifting(video_path):

    # Load YOLOv8 model
    model_yolo = YOLO(
        r'D:\SUSPICIOUS-ACTIVITY-DETECTION-main\SUSPICIOUS-ACTIVITY-DETECTION-main\yolo11_suspicious_activity-main\yolo11s-pose.pt'
    )

    # Load trained XGBoost model
    model = xgb.Booster()
    model.load_model(
        r'D:\SUSPICIOUS-ACTIVITY-DETECTION-main\SUSPICIOUS-ACTIVITY-DETECTION-main\yolo11_suspicious_activity-main\trained_model.json'
    )

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_tot = 0
    count = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Warning: Frame could not be read.")
            break

        count += 1

        if count % 3 != 0:
            continue

        # Resize frame
        frame = cv2.resize(frame, (1018, 600))

        # Run YOLO
        results = model_yolo(frame, verbose=False)

        # Annotated frame
        annotated_frame = results[0].plot(boxes=False)

        for r in results:

            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

            for index, box in enumerate(bound_box):

                if conf[index] > 0.55:

                    x1, y1, x2, y2 = box.tolist()

                    # Prepare dataframe
                    data = {}

                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    df = pd.DataFrame(data, index=[0])

                    dmatrix = xgb.DMatrix(df)

                    # Prediction
                    sus = model.predict(dmatrix)

                    binary_predictions = (sus > 0.5).astype(int)

                    print(f'Prediction: {binary_predictions}')

                    # Suspicious
                    if binary_predictions == 0:

                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 255),
                            2
                        )

                        cvzone.putTextRect(
                            annotated_frame,
                            "Suspicious",
                            (int(x1), int(y1) + 50),
                            1,
                            1
                        )

                    # Normal
                    else:

                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2
                        )

                        cvzone.putTextRect(
                            annotated_frame,
                            "Normal",
                            (int(x1), int(y1) + 50),
                            1,
                            1
                        )

        # Display frame
        cv2.imshow('Frame', annotated_frame)

        # Quit with q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Run function
detect_shoplifting(video_path)