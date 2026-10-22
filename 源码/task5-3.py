from ultralytics import YOLO

path = r"C:\Users\mrxj2\Downloads\drone and bird.v2i.yolov11\data.yaml"


model = YOLO(r"C:\Users\mrxj2\Downloads\drone.pt")

train_results = model.train(
    data=path,
    epochs=5,
    name="task3",
)

print("save", model.trainer.save_dir)