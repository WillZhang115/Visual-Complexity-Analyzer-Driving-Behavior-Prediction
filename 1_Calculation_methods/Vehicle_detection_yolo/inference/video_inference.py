import cv2
from ultralytics import YOLO

def detect_video(input_path, output_path):
    # Load the model
    model = YOLO('../runs/detect/train/weights/best.pt')
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, 
                         (frame_width, frame_height))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Perform inference
        results = model(frame)
        
        # Draw results
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    print(f"Video result saved to {output_path}")

if __name__ == '__main__':
    detect_video(
        input_path='../data/examples/test_video.mov',
        output_path='../runs/detect/output/video_result.mov'
    )
