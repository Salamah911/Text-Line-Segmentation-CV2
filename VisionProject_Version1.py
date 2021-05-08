#!/usr/bin/python
from cv2 import cv2
import numpy as np
from random import randint
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import webbrowser

while 1:
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    imgname = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    # initialdir="/home/salamah/Documents/Computer Vision project/ahte_dataset/ahte_test_binary_images/"
    if imgname == "" or imgname == () or imgname == None or imgname == []:
        exit(0)

    image = cv2.imread(imgname)
    imgcpy = image.copy()
    imgcpy = cv2.resize(imgcpy , (imgcpy.shape[1],imgcpy.shape[0]*3))
    gray = cv2.cvtColor(imgcpy, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("v1_out/gray.png", gray)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    cv2.imwrite("v1_out/blur.png", blur)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 400:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
    cv2.imwrite("v1_out/thresh.png", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,5))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite("v1_out/close.png", close)
    cv2.imwrite("v1_out/thresh_v1.png", thresh)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,1))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    cv2.imwrite('v1_out/close_v1.png',close)

    kernel = np.ones((1,60), np.uint8)
    close = cv2.erode(close,kernel,iterations=4)	
    cv2.imwrite('v1_out/close_v2.png',close)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 300:
            cv2.drawContours(close, [c], -1, (0,0,0), -1)


    cv2.imwrite("v1_out/close_v3.png", close)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
    close =  cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=3)
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10000:
            cv2.drawContours(close, [c], -1, (0,0,0), -1)

    cv2.imwrite("v1_out/close_v4.png", close)

    kernel = np.ones((1,30), np.uint8)
    close = cv2.erode(close,kernel,iterations=2)	
    cv2.imwrite('v1_out/close_v5.png',close)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (140,1))
    close = cv2.dilate(close,kernel, iterations=2)
    cv2.imwrite("v1_out/close_v6.png", close)
    
    close = cv2.resize(close , (imgcpy.shape[1],image.shape[0]))
    imgcpy = cv2.resize(imgcpy , (imgcpy.shape[1],image.shape[0]))


    (contours, _) = cv2.findContours(close, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(imgcpy,(x-1,y-5),(x+w,y+h),(randint(0, 255),randint(0, 255),randint(0, 255)),5)
    
    
   
    cv2.imwrite("v1_out/imgContoure.png", imgcpy)
    webbrowser.open("v1_out/imgContoure.png")
   
