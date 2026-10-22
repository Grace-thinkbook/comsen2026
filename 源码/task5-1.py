from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
    source="C:/Users/mrxj2/Desktop/example",
    imgsz=640,
    conf=0.25,
    save=True,
    show=True   
)


print(f"save{model.predictor.save_dir}")


for r in results:
    boxes = r.boxes
    print(f"path{r.path}")
    print(f"number{len(boxes)}")
    for box in boxes:
        name = model.names[int(box.cls)]
        number = float(box.conf)
        print(f"    {name} confidence{number:.2f}")