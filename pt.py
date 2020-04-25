#!/usr/bin/python3
# programing by M.kawase(Embed A.I. Labo.)
#

import sys
import numpy as np
import cv2

files = sys.argv[1:]
ec=0
fn=len(files)
pos=np.array([[0,0],[0,0]])
nn=2

def msevent(ev, x, y, fl, prm):
    global ec,pos
    if ev == cv2.EVENT_LBUTTONUP:
        pos[ec, 0],pos[ec, 1] = (x,y)
        ec = ec + 1
        print('input No{} (x,y)=({},{})'.format(ec,x,y))

if fn > 0:
    img=cv2.imread(files[0])
    y,x,d=img.shape
    img2 = cv2.resize(img , (int(x*(1/nn)), int(y*(1/nn))))
    print(img2.shape)
    cv2.imshow('T01',img2)
    cv2.setMouseCallback('T01',msevent)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    wx = pos[1,0]-pos[0,0]
    wy = pos[1,1]-pos[0,1]
    ty = tx = int(np.sqrt(fn)+1)
    cvs = np.array([[[0] * 3] * (tx * wx)] * (ty * wy),dtype=img2.dtype)
    for p in range(fn):
        img=cv2.imread(files[p])
        y,x,d=img.shape
        img2 = cv2.resize(img , (int(x*(1/nn)), int(y*(1/nn))))
        print(img2.shape)
        xx = (p % tx) * wx
        yy = (p // ty) * wy
        cvs[yy:(yy+wy),xx:(xx+wx),:]=img2[pos[0,1]:pos[1,1],pos[0,0]:pos[1,0],:]
        cv2.imshow('T01',img2[pos[0,1]:pos[1,1],pos[0,0]:pos[1,0],:])
        cv2.waitKey(0)
    cv2.imshow('T01',cvs)
    cv2.waitKey(0)
    cv2.imwrite('./pictile.jpg',cvs)
else:
    print('No files')
