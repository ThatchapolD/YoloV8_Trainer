from multiprocessing import freeze_support
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8m.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8m.pt')

if __name__ == '__main__':
    freeze_support()


    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='D:\Yolov8\Dataset\datacolab.yaml', epochs=2, imgsz= 740)

    # Evaluate the model's performance on the validation set
    results = model.val()