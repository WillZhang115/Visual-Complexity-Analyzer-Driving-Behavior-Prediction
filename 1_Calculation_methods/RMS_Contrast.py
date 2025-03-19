import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog

def compute_rms_contrast(image):
    """Calculate the RMS contrast metric for an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a local contrast neighborhood
    kernel_size = (15, 15)
    local_std = cv2.boxFilter(gray.astype(np.float32), -1, kernel_size, normalize=False)
    
    V = local_std.flatten()
    M = np.mean(V)
    S = np.std(V)
    
    return M * S

def process_images(folder):
    """Process all images in the folder and calculate their RMS contrast."""
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    rms_contrast_results = []
    
    for image_name in image_files:
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        contrast = compute_rms_contrast(image)
        rms_contrast_results.append((image_name, contrast))
    
    return rms_contrast_results

def process_videos(folder):
    """Process all videos in the folder and calculate their average RMS contrast."""
    video_files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    rms_contrast_results = []
    
    for video_name in video_files:
        video_path = os.path.join(folder, video_name)
        cap = cv2.VideoCapture(video_path)
        
        total_contrast = 0
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            total_contrast += compute_rms_contrast(frame)
            frame_count += 1
        
        cap.release()
        
        avg_contrast = total_contrast / frame_count if frame_count > 0 else 0
        rms_contrast_results.append((video_name, avg_contrast))
    
    return rms_contrast_results

def save_results(folder, image_results, video_results):
    """Save RMS contrast results to an Excel file."""
    output_path = os.path.join(folder, "RMS_Contrast_Results.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        if image_results:
            df_images = pd.DataFrame(image_results, columns=["Image Name", "RMS Contrast"])
            df_images.to_excel(writer, sheet_name="Images", index=False)
        
        if video_results:
            df_videos = pd.DataFrame(video_results, columns=["Video Name", "Average RMS Contrast"])
            df_videos.to_excel(writer, sheet_name="Videos", index=False)
    
    print(f"Results saved to {output_path}")


