import cv2

def a(b,c,d=0):
    e=cv2.VideoCapture(b)
    f=e.get(cv2.CAP_PROP_FPS)
    g=int(e.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(e.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i=cv2.VideoWriter_fourcc(*'mp4v')
    j=cv2.VideoWriter(c,i,f,(g,h))
    print("skip",d,"frames")
    for _ in range(d):
        e.read()
    ret,k=e.read()
    if not ret:
        e.release()
        j.release()
        cv2.destroyAllWindows()
        return
    l=cv2.selectROI("select",k,False)
    if l[2]<=0 or l[3]<=0:
        e.release()
        j.release()
        cv2.destroyAllWindows()
        return
    m=cv2.TrackerCSRT_create()
    m.init(k,l)
    while e.isOpened():
        ret,k=e.read()
        n,o=m.update(k)
        if n:
            x,y,w,h=o
            cv2.rectangle(k,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
        else:
            cv2.putText(k,'lost',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        j.write(k)
        cv2.imshow("preview",k)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    e.release()
    j.release()
    cv2.destroyAllWindows()
    print("done")

p=r"C:\Users\mrxj2\Desktop\example3.mp4"
q=r"C:\Users\mrxj2\Desktop\track_result3.mp4"
a(p,q,0)
cv2.destroyAllWindows()
input()
r=r"C:\Users\mrxj2\Desktop\example4.mp4"
s=r"C:\Users\mrxj2\Desktop\track_result4.mp4"
a(r,s,25)
print("done")