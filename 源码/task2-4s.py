import cv2
import numpy as np

a = r"C:\Users\mrxj2\Desktop\example2.mp4"
b = r"C:\Users\mrxj2\Desktop\result_frame_diff_color.mp4"
c = r"C:\Users\mrxj2\Desktop\result_frame_diff_mask.mp4"

d = cv2.VideoCapture(a)
e = d.get(cv2.CAP_PROP_FPS)
f = int(d.get(cv2.CAP_PROP_FRAME_WIDTH))
g = int(d.get(cv2.CAP_PROP_FRAME_HEIGHT))

h = cv2.VideoWriter_fourcc(*'mp4v')
i = cv2.VideoWriter(b, h, e, (f, g))
j = cv2.VideoWriter(c, h, e, (f, g), False)

ret, k = d.read()
if not ret:
    exit()

l = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)

while d.isOpened():
    ret, m = d.read()
    if not ret:
        break

    n = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    o = cv2.absdiff(l, n)
    _, p = cv2.threshold(o, 25, 255, cv2.THRESH_BINARY)
    q = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    r = cv2.morphologyEx(p, cv2.MORPH_OPEN, q)

    s, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for t in s:
        u = cv2.contourArea(t)
        if u > 80:
            cv2.drawContours(m, [t], -1, (0, 255, 0), 2)
    
    i.write(m)
    j.write(r)

    cv2.imshow("img", m)
    cv2.imshow("mask", r)

    l = n.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

d.release()
i.release()
j.release()
cv2.destroyAllWindows()
print("done")