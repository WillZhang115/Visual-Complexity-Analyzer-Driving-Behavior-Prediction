import os
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# 1. Load YOLOv8 Model
# -------------------------------
# Set the generic model directory and file path
model_dir = os.path.join('path', 'to', 'model')  # Update with your model folder path
model_path = os.path.join(model_dir, 'best.pt')    # Your YOLOv8 model file

# Load the YOLOv8 model
model = YOLO(model_path)

# -------------------------------
# 2. Setup Video Directory and File List
# -------------------------------
# Set the generic video directory path
video_dir = os.path.join('path', 'to', 'videos')   # Update with your video folder path

# Retrieve all .mp4 files in the directory
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
video_files.sort()  # Optional: sort files by name

# -------------------------------
# 3. Process Each Video and Count Cars per Frame
# -------------------------------
# Dictionary to store per-frame car counts for each video
video_frame_counts = {}

# Loop through each video file
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        continue
    
    frame_counts = []  # List to store detection count for each frame
    frame_idx = 0      # Frame counter
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_idx += 1
        
        # Convert the frame from BGR (OpenCV format) to RGB (model format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection on the current frame
        results = model(rgb_frame)
        
        # Get the number of detected objects in the frame.
        # (Assumes that all detections correspond to cars. If needed, filter by class label.)
        detection_count = results[0].boxes.shape[0]
        frame_counts.append(detection_count)
        
        # Optional: To display detection results, uncomment the following lines.
        # for box in results[0].boxes:
        #     x1, y1, x2, y2 = map(int, box.xyxy[0])
        #     conf = box.conf[0]
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.imshow("Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    video_frame_counts[video_file] = frame_counts
    print(f"Processed '{video_file}' with {frame_idx} frames.")

# -------------------------------
# 4. Calculate Total Car Counts per Video
# -------------------------------
# Compute the total count (sum of detections over all frames) for each video
video_total_counts = {video: sum(counts) for video, counts in video_frame_counts.items()}

print("\nTotal car counts per video:")
for video, total in video_total_counts.items():
    print(f"{video}: {total}")

# -------------------------------
# 5. Save the Results to Files
# -------------------------------
# Set a generic output directory path and create it if it does not exist
output_dir = os.path.join('path', 'to', 'output')   # Update with your desired output folder path
os.makedirs(output_dir, exist_ok=True)

# Save per-frame counts for each video as a CSV file
for video, counts in video_frame_counts.items():
    output_file = os.path.join(output_dir, f"{os.path.splitext(video)[0]}_frame_counts.csv")
    np.savetxt(output_file, counts, delimiter=",", fmt='%d')
    print(f"Saved per-frame counts for '{video}' to {output_file}")

# Save total counts for all videos to a CSV file
total_counts_file = os.path.join(output_dir, "video_total_counts.csv")
with open(total_counts_file, 'w') as f:
    f.write("video,total_count\n")
    for video, total in video_total_counts.items():
        f.write(f"{video},{total}\n")
print(f"Saved total counts to {total_counts_file}")
