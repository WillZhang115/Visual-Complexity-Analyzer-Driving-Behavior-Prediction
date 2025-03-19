import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tkinter import Tk, filedialog

def calculate_video_ssim(video_path):
    """Calculate the total SSIM value for a given video."""
    cap = cv2.VideoCapture(video_path)
    success, prev_frame = cap.read()
    
    if not success:
        print(f"Failed to read {video_path}")
        return 0

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_ssim = 0
    frame_count = 0

    while True:
        success, current_frame = cap.read()
        if not success:
            break

        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(prev_frame, current_frame, full=True)
        total_ssim += score
        frame_count += 1
        prev_frame = current_frame

    cap.release()
    return total_ssim

def process_videos(folder):
    """Process all videos in the folder and calculate SSIM values."""
    video_files = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
    ssim_results = []

    for video_name in video_files:
        video_path = os.path.join(folder, video_name)
        ssim_value = calculate_video_ssim(video_path)
        ssim_results.append((video_name, ssim_value))
        print(f"SSIM for {video_name}: {ssim_value:.2f}")

    # Save results to an Excel file
    df = pd.DataFrame(ssim_results, columns=["Video Name", "SSIM Value"])
    output_path = os.path.join(folder, "SSIM_all.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")
