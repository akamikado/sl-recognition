from ultralytics import YOLO

model = YOLO('../weights/yolov8n.pt')

results = model.train(data="../datasets/hand_dataset.yaml", epochs=3,
                      verbose=True, batch=1, optimizer='Adam')
