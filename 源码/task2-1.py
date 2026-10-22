import cv2
from PIL import Image
import numpy as np

a = Image.open(r"C:\Users\mnxj2\Desktop\example1.jpg")


array = np.array(a)
cv = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

b, g, r = cv2.split(cv)

cv2.imwrite("B.jpg", b)
cv2.imwrite("G.jpg", g)
cv2.imwrite("R.jpg", r)


cv2.imshow("rare", cv)
cv2.imshow("Blue", b)
cv2.imshow("Green", g)
cv2.imshow("Red", r)


cv2.waitKey(0)
cv2.destroyAllWindows()