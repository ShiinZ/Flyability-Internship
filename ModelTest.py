from ultralytics import YOLO

def TestModel(ModelPath, ImagePath):
    model = YOLO(ModelPath)

    results = model.predict(source=ImagePath, conf=0.25)
    
    # Position of the detected box
    for result in results:
        print(f"Detections for image: {ImagePath}")
        
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            
            print(f"Class ID: {class_id}, Box: ({x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f})")

        detected_image_path = result.save()
        print(f"Image saved in : {detected_image_path}")


ModelFolderPath = "Models/FinetunedModel/"

#Change the model here
ModelPath = "saved_modelNanoYolo30e.pt"

# Change the image Here
ImagePath = "Dataset\Images\Valid\image11.jpg"
    
TestModel(ModelFolderPath + ModelPath, ImagePath)