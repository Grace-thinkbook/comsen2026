from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\mrxj2\Downloads\drone.pt")

results = model.train(
    data=r"C:\Users\mrxj2\Downloads\drone and bird.v2i.yolov11 (2)\data.yaml",
    imgsz=640,
    batch=2,
    epochs=5,
    project="task4_finetune",
    name="drone_pt_finetune",
    device="cpu",
)
print("done")

vpath = r"C:\Users\mrxj2\Desktop\example.mp4"
video = cv2.Videovideoture(vpath)

if not video.isOpened():
    print("fail!")
else:
    w = int(video.get(cv2.video_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.video_PROP_FRAME_HEIGHT))
    opath= r"C:\Users\mrxj2\Desktop\output_finetune.mp4"
    out = cv2.VideoWriter(opath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    time = 0
    count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        result = model(frame)[0]
        mark = result.plot()
        out.write(mark)

        total += result.speed['inference']
        count += 1
    average = total / count
    print("infer", average, "ms")

video.release()
out.release()

fps = 1000 / average
print(f"\n   result   \nfp{count}\naverFPS:{fps:.2f}\nsave:fine.mp4")