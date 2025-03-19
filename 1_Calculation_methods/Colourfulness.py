import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog

def compute_colorfulness(image):
    """Calculate the colorfulness metric for an image."""
    R, G, B = cv2.split(image.astype("float"))

    rg = R - G
    yb = 0.5 * (R + G) - B

    Srg = np.std(rg)
    Syb = np.std(yb)
    Srgyb = np.sqrt(Srg**2 + Syb**2)

    Mrg = np.mean(rg)
    Myb = np.mean(yb)
    Mrgyb = np.sqrt(Mrg**2 + Myb**2)

    return Srgyb + 0.3 * Mrgyb

def process_images(folder):
    """Process all images in the folder and calculate their colorfulness."""
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    total_colorfulness = 0

    for image_name in image_files:
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        C = compute_colorfulness(image)
        total_colorfulness += C
    
    print(f"Total colorfulness of all images: {total_colorfulness:.2f}")

def process_videos(folder):
    """Process all videos in the folder and calculate their colorfulness."""
    video_files = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
    colorfulness_results = []

    for video_name in video_files:
        video_path = os.path.join(folder, video_name)
        cap = cv2.VideoCapture(video_path)

        total_colorfulness = 0
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_colorfulness += compute_colorfulness(frame)
            frame_count += 1

        cap.release()

        colorfulness_results.append((video_name, total_colorfulness))
        print(f"Colorfulness of video {video_name}: {total_colorfulness:.2f}")

    # Save results to an Excel file
    df = pd.DataFrame(colorfulness_results, columns=["Video Name", "Colorfulness"])
    output_path = os.path.join(folder, "Color_all.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


