from ultralytics import YOLO
model = YOLO('./best.pt')
# Define path to the image file
# source = file_path
source = 'xteeth.jpg'
# Run inference on the source
# results = model(source)  # list of Results objects
model.predict(source, save=True, imgsz=640)