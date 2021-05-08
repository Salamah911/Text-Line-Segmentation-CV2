#!/usr/bin/python

from cv2 import cv2
import numpy as np
from random import randint
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import webbrowser
import seam_carving
from PIL import Image
import seam_carver
import os
import PIL
import glob

def removeNoise(img, AreaToRemove):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < AreaToRemove:
            cv2.drawContours(img, [c], -1, (0,0,0), -1)

def drawContours(source, dist):
    (contours, _) = cv2.findContours(source, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15000:
            x,y,w,h = cv2.boundingRect(cnt)
            x += 7
            w -= 13
            y -= 10
            h += 15
            cv2.rectangle(dist,(x-1,y-5),(x+w,y+h),(randint(0, 255),randint(0, 255),randint(0, 255)),5)
    cv2.imwrite("v2_out/imgContoure.png", dist)
    webbrowser.open("v2_out/imgContoure.png")

while 1:
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    imgname = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    # initialdir="/home/salamah/Documents/Computer Vision project/ahte_dataset/ahte_test_binary_images/"
    if imgname == "" or imgname == () or imgname == None or imgname == []:
        exit(0)

    image = cv2.imread(imgname)
    imgcpy = image.copy()
    cv2.imwrite("v2_out/WorkingImage.png", imgcpy)
    imgcpy = cv2.resize(imgcpy, (imgcpy.shape[1], imgcpy.shape[0]*3))
    gray = cv2.cvtColor(imgcpy, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("v2_out/gray.png", gray)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # thresh = cv2.resize(thresh, (thresh.shape[1], thresh.shape[0]*3))

    removeNoise(thresh , 300)
    cv2.imwrite("v2_out/thresh.png", thresh)

    kernel = np.ones((1,5), np.uint8)
    close = cv2.erode(thresh,kernel,iterations=2)	
    
    removeNoise(close , 100)

    cv2.imwrite('v2_out/close.png',close)

    blur = cv2.blur(close, (150, 1))
    # thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite("v2_out/blur.png", blur)
    # cv2.imwrite("FinalOutput/thresh_v1.png", thresh)


    removeNoise(blur, 100)

    # src_h, src_w = blur.shape
    # dst = seam_carving.resize(
    # blur, (src_w, src_h + 1000),
    # energy_mode='forward',   # Choose from {backward, forward}
    # order='height-first',  # Choose from {width-first, height-first}
    # keep_mask=None
    # )
    # # Image.fromarray(dst).show()
    # cv2.imwrite("FinalOutput/seam.png", dst)


    thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("v2_out/thresh_v1.png", thresh)
    
    removeNoise(thresh, 1000)
    cv2.imwrite("v2_out/thresh_v2.png", thresh)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    
    kernel = np.ones((2,10), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=3)
    cv2.imwrite("v2_out/erosion.png", erosion)

    kernel = np.ones((5,1), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    cv2.imwrite("v2_out/dilation.png", dilation)

    blur = cv2.blur(dilation, (500, 1))
    cv2.imwrite("v2_out/blur_v1.png", blur)

    kernel = np.ones((10,1), np.uint8)
    erosion = cv2.erode(blur, kernel)
    cv2.imwrite("v2_out/erosion_vblur.png", erosion)


    thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("v2_out/thresh_v3.png", thresh)
    thresh = cv2.resize(thresh, (image.shape[1], image.shape[0]))
    imgcpy = image.copy()
    drawContours(thresh, imgcpy)


    # kernel = np.ones((1,80), np.uint8)
    # erosion = cv2.erode(thresh, kernel, iterations=4)
    # cv2.imwrite("FinalOutput/erosion_v1.png", erosion)

    # kernel = np.ones((8,1), np.uint8)
    # dilation = cv2.dilate(erosion, kernel, iterations=2)
    # removeNoise(dilation, 2000)

    # cv2.imwrite("FinalOutput/dilation_v1.png", dilation)
    # dilation = cv2.resize(dilation, (image.shape[1], image.shape[0]))
    # imgcpy = image.copy()
    # drawContours(dilation, imgcpy)

