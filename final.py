import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import sys
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
Kernal = np.ones((10, 10), np.uint8)
cap=cv2.VideoCapture(0)
y1=[]
x1=[]
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
im = np.zeros((h,w, 3), dtype = "uint8")+255
f=0
while(1):
    if keyboard.is_pressed('c'):
        cv2.destroyAllWindows()
        cap.release()
        sys.exit()
    ret,frame=cap.read()
    frame = cv2.flip(frame, +1)
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lb = np.array([22,0,171])               
    ub = np.array([245, 245, 245])
    mask = cv2.inRange(frame2, lb, ub)
    mas = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernal)
    #cv2.imshow('mask',mas)
    res = cv2.bitwise_and(frame, frame, mask = mas)
    #cv2.imshow('res',res)
    if not (keyboard.is_pressed('ctrl')):
        contours, hierarchy = cv2.findContours(mas, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x = int(x)
            y = int(y)
            x1.append(x)
            y1.append(y)
    for i in range(len(x1)):
        cv2.circle(frame, (x1[i], y1[i]), 0, (0, 0, 255), 6)
        cv2.circle(im, (x1[i], y1[i]), 0, (0,0, 0), 6)        
    cv2.imshow('frame',frame)
    #cv2.imshow('im', im)
    if cv2.waitKey(1)  == ord('q'):
        break
    cv2.imwrite('out.jpg',im)
cv2.destroyAllWindows()
cap.release()
X=cv2.imread('out.jpg')
X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
(thresh, X) = cv2.threshold(X, 127, 255, cv2.THRESH_BINARY)
f=0
jl=[]
il=[]
#print(X)
for i in range(X.shape[0]):
    t=X[i]
    for j in range(X.shape[1]):
        if(t[j]==0):
            il.append(i)
            break
            
##print(X.shape)
#print(il)
Y=X[il[0]-5:il[len(il)-1]+5]
for j in range(Y.shape[1]):
    t=Y[:,j]
    for i in range(Y.shape[0]):
        if(t[i]==0):
            jl.append(j)
            break
jll=[]
jll.append(jl[0])
for i in range(1,len(jl)):
    if(abs(jl[i]-jl[i-1])>3):
        jll.append(jl[i-1])
        jll.append(jl[i])
jll.append(jl[len(jl)-1])
##print(jll)
##print(jl)
##print(len(jll))
im=[]
for i in range(int(len(jll)/2)):
    im.append(Y[:,jll[2*i]-5:jll[2*i+1]+5])
##print(len(im))
for i in range(len(im)):
    #cv2.imshow(f'im{i}',im[i])
    cv2.imwrite(f'im{i}.jpg',im[i])
##X,Y = pickle.load(open('tam.pickle','rb'))
##X=X.reshape(-1,100,100,1)
##Y=np.array(Y).reshape(-1,1)
d=['\u0B95','\u0B99','\u0B9A','\u0B9E','\u0B9F','\u0BA3','\u0BA4','\u0BA8',
   '\u0BAA','\u0BAE','\u0BAF','\u0BB0','\u0BB2','\u0BB5','\u0BB4','\u0BB3','\u0BB1','\u0BA9']
model=load_model('fin.h5')
for i in range(len(im)):
    ar=f'im{i}.jpg'
    x=cv2.imread(ar)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x=cv2.resize(x,(100,100))
    (thresh, x) = cv2.threshold(x, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    x=x/255
    x=x.astype('uint8')
    x=np.array(x).reshape(1,100,100,1)
    p=model.predict([x])
    p=np.ndarray.tolist(p.reshape(-1,1))
    #print(p)
    i=p.index(max(p))
    print(d[i])
