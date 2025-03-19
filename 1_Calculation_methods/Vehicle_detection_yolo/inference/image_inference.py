import cv2
from ultralytics import YOLO

def detect_image(input_path, output_path):
    # Load the trained model
    model = YOLO('../runs/detect/train/weights/best.pt')
    
    # Perform inference
    results = model(input_path)
    
    # Save results
    for r in results:
        im_array = r.plot()
        cv2.imwrite(output_path, im_array)
        print(f"Result saved to {output_path}")

if __name__ == '__main__':
    detect_image(
        input_path='../data/examples/test_image.jpg',
        output_path='../runs/detect/output/image_result.jpg'
    )
