import cv2,time
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

def g(m,v,pt=0):
    t=0
    if pt:
        m=YOLO(m)
        c=cv2.VideoCapture(v)
        while c.isOpened():
            r,f=c.read()
            if not r:break
            s=time.time()
            m.predict(f,imgsz=(640,640),save=0,verbose=0)
            t+=time.time()-s
        c.release()
    else:
        s=ort.InferenceSession(m,providers=["CPUExecutionProvider"])
        i=s.get_inputs()[0].name
        c=cv2.VideoCapture(v)
        while c.isOpened():
            r,f=c.read()
            if not r:break
            x=cv2.resize(f,(640,640))
            x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
            x=np.transpose(x,(2,0,1))
            x=np.expand_dims(x,0).astype(np.float32)/255
            st=time.time()
            s.run(None,{i:x})
            t+=time.time()-st
        c.release()
    return t