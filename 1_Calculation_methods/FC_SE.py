import os
import cv2
import numpy as np
import pandas as pd
from skimage import color
from skimage.metrics import structural_similarity as ssim
from moviepy.editor import VideoFileClip

def compute_clutter(image):
    # Placeholder function to compute clutter in color, contrast, and orientation
    # You need to implement this function based on the MATLAB code
    color_clutter = np.random.random(image.shape[:2])  # Replace with actual computation
    contrast_clutter = np.random.random(image.shape[:2])  # Replace with actual computation
    orient_clutter = np.random.random(image.shape[:2])  # Replace with actual computation
    return color_clutter, contrast_clutter, orient_clutter

def get_clutter_fc(image, p=1):
    # Compute clutter in color, contrast, and orientation
    color_clutter, contrast_clutter, orient_clutter = compute_clutter(image)
    
    # Combine color, contrast, and orientation clutter maps
    clutter_map_fc = (color_clutter / 0.2088 + contrast_clutter / 0.0660 + orient_clutter / 0.0269)
    
    # Compute global clutter using Minkowski mean
    clutter_scalar_fc = np.mean(clutter_map_fc ** p) ** (1 / p)
    
    return clutter_scalar_fc, clutter_map_fc

def band_entropy(image, wlevels=3, wor=4):
    # Placeholder function to compute band entropy for an image
    # Implement based on MATLAB code
    return np.random.random(wlevels)  # Replace with actual entropy calculation

def get_clutter_se(image, wlevels=3, wght_chrom=0.0625):
    # Convert image to Lab color space if it is a color image
    if len(image.shape) == 3:
        image_lab = color.rgb2lab(image)
    else:
        image_lab = image
    
    # Compute entropy for the luminance channel
    en_band = band_entropy(image_lab[:, :, 0], wlevels)
    clutter_se = np.mean(en_band)
    
    # Compute entropy for chrominance channels if present
    if len(image.shape) == 3:
        for i in range(1, 3):
            chrom_channel = image_lab[:, :, i]
            if np.max(chrom_channel) - np.min(chrom_channel) < 0.008:
                chrom_channel = np.zeros_like(chrom_channel)
            en_band = band_entropy(chrom_channel, wlevels)
            clutter_se += wght_chrom * np.mean(en_band)
    
    clutter_se /= (1 + 2 * wght_chrom)
    return clutter_se

def process_videos(folder_path):
    # Initialize storage for Feature Congestion and Subband Entropy values
    num_videos = 29
    max_frames = 601
    fc_values = np.zeros((num_videos, max_frames))
    se_values = np.zeros((num_videos, max_frames))
    
    for i in range(1, num_videos + 1):
        video_path = os.path.join(folder_path, f"{i}.mp4")
        if not os.path.exists(video_path):
            continue
        
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for r in range(min(frame_count, max_frames)):
            ret, frame = video.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute Feature Congestion
            fc, _ = get_clutter_fc(gray_frame)
            fc_values[i - 1, r] = fc
            
            # Compute Subband Entropy
            se = get_clutter_se(gray_frame)
            se_values[i - 1, r] = se
        
        video.release()
    
    # Compute mean and variance for Feature Congestion and Subband Entropy
    fc_mean = np.mean(fc_values, axis=1)
    fc_var = np.var(fc_values, axis=1)
    se_mean = np.mean(se_values, axis=1)
    se_var = np.var(se_values, axis=1)
    
    # Save results to CSV files
    fc_df = pd.DataFrame(fc_values)
    fc_df['Mean'] = fc_mean
    fc_df['Variance'] = fc_var
    fc_df.to_csv(os.path.join(folder_path, 'Feature_Congestion.csv'), index=False)
    
    se_df = pd.DataFrame(se_values)
    se_df['Mean'] = se_mean
    se_df['Variance'] = se_var
    se_df.to_csv(os.path.join(folder_path, 'Subband_Entropy.csv'), index=False)
    
