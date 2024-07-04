from ultralytics import YOLO

model = YOLO("yolov8x-cls.pt")

results = model.train(data="", epochs=, imgsz=)
