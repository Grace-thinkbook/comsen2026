import cv2 as cv
import numpy as np

def bound(img,c,i):
    x,y,w,h = cv.boundingRect(c[i])
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

def area(img,c,i):
    rect = cv.minAreaRect(c[i])
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img,[box],0,(0,255,0),2)


img = cv.imread("example1.jpg")
imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(imgG, (3,3), 0)
B = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11, 2)
c,a = cv.findContours(B,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

tarea= 0
tp = 0
for i in range(len(c)):
    area = cv.contourArea(c)
    peri = cv.arcLength(c, True)
    if peri == 0:
        continue
    cir = 4 * np.pi * area / (peri ** 2)
    if area > 200 and cir > 0.15:
        tarea+= area
        tp += peri
        cv.drawContours(img, [c], -1, (0,255,0), 2)
        bound(img,c,i)
        area(img,c,i)
print(f"s = {tarea:.2f}")
print(f"c = {tp:.2f}")
cv.imwrite("result.jpg", img)
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
