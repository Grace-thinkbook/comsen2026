import cv2
import numpy as np
from PIL import Image

path = r"C:\Users\mnxj2\Desktop\example1.jpg"
a = Image.open(path)
img = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2BGR)

h, w = img.shape[:2]

resize = cv2.resize(img, None, fx=0.6, fy=0.6)

fanzhuan = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
fanzhuans = cv2.warpAffine(img, fanzhuan, (w, h))

tran = np.float32([[1, 0, 100],
                         [0, 1, 50]])
trans = cv2.warpAffine(img, tran, (w, h))

flip = cv2.flip(img, 1)

cv2.imwrite("resized.jpg", resize)
cv2.imwrite("fanzhuan.jpg", rot)
cv2.imwrite("translated.jpg", trans)
cv2.imwrite("flipped.jpg", flip)

cv2.imshow("rare", img)
cv2.imshow("resize", resize)
cv2.imshow("f", fanzhuans)
cv2.imshow("trans", trans)
cv2.imshow("flip", flip)

cv2.waitKey(0)
cv2.destroyAllWindows()