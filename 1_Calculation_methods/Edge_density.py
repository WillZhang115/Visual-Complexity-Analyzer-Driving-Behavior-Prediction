import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog

def compute_edge_density(image):
    """Calculate the edge density of an image using the Canny filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 28, 69)  # Scaled [0.11*255, 0.27*255]
    total_pixels = edges.size
    edge_pixels = np.sum(edges > 0)
    return edge_pixels / total_pixels, edges

def process_images(folder):
    """Process all images in the folder and calculate their edge density."""
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    for image_name in image_files:
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        edge_density, edges = compute_edge_density(image)
        print(f"Edge Density for {image_name}: {edge_density:.4f}")
        output_path = os.path.join(folder, f"Canny_{image_name}")
        cv2.imwrite(output_path, edges)

def process_videos(folder):
    """Process all videos in the folder and apply Canny filter to frames."""
    video_files = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
    edge_densities = []
    
    for video_name in video_files:
        video_path = os.path.join(folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        output_video_path = os.path.join(folder, f"Canny_{video_name}")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, 
                              (int(cap.get(3)), int(cap.get(4))), isColor=False)
        
        total_edge_density = 0
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            edge_density, edges = compute_edge_density(frame)
            total_edge_density += edge_density
            
            output_image_path = os.path.join(folder, f"Canny_{video_name}_{frame_idx}.jpg")
            cv2.imwrite(output_image_path, edges)
            out.write(edges)
        
        cap.release()
        out.release()
        edge_densities.append((video_name, total_edge_density))
        print(f"Total Edge Density for {video_name}: {total_edge_density:.4f}")
    
    # Save edge density results to an Excel file
    df = pd.DataFrame(edge_densities, columns=["Video Name", "Total Edge Density"])
    output_path = os.path.join(folder, "Edge_Density_Results.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

