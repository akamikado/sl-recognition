from model import ModifiedYolo
# from reduce_size_of_dataset import split_dataset
#
# split = float(input("Enter the split: "))
# split_dataset(split)

model = ModifiedYolo(
    "resnet-yolo.yaml").load("yolov8n-cls.pt")

results = model.train(data=f"isl_dataset-12",
                      epochs=3, verbose=True, batch=1, optimizer='Adam', lr0=0.00050)
