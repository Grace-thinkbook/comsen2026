import cv2
import numpy as np

v = cv2.VideoCapture("test.mp4")
s = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

fcc = cv2.VideoWriter_fourcc(*"mp4v")
c = cv2.VideoWriter("outc.mp4", fcc, 30, (640,480))
m = cv2.VideoWriter("outm.mp4", fcc, 30, (640,480), isColor=False)

k1 = np.ones((2,2), np.uint8)
k2 = np.ones((3,3), np.uint8)

while True:
    ret, frm = v.read()
    
    frm = cv2.resize(frm, (640,480))
    mk = s.apply(frm)
    
    mk[mk == 127] = 0
    _, mk = cv2.threshold(mk, 220, 255, cv2.THRESH_BINARY)

    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, k1)
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, k2)
    mk = cv2.morphologyEx(mk, cv2.MORPH_DILATE, k2)

    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for t in cnts:
        ar = cv2.contourArea(t)
        if ar > 120:
            cv2.drawContours(frm, [t], -1, (0,255,0), 2)
    
    c.write(frm)
    m.write(mk)

    cv2.imshow("mask", mk)
    cv2.imshow("frame", frm)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

v.release()
c.release()
m.release()
cv2.destroyAllWindows()
print("success")