from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

results = model.train(data="asl_alphabet_train", epochs=300, imgsz=200)
