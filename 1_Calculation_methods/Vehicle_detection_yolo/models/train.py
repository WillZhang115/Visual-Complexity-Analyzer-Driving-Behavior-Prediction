from ultralytics import YOLO

def train_model():
    # Load a pre-trained model
    model = YOLO('yolov8s.pt')
    
    # Training configuration
    results = model.train(
        data='../data/datasets/vehicles.yaml',  # Dataset configuration file
        epochs=50,                             # Number of training epochs
        batch=16,                               # Batch size
        imgsz=640,                              # Input image size
        save=True,                              # Save the model
        save_period=5,                          # Models saved every 5 epochs点击并应用
        device='0',                             # Use GPU (set to 'cpu' to use CPU)
        project='../runs/detect',               # Directory to save training results
        name='train'                            # Name of the training run
    )

if __name__ == '__main__':
    train_model()
