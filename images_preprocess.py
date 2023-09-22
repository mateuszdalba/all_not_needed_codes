import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir


for folder in os.listdir("/CartingVision/images/"):
    folder_dir = f"/CartingVision/images/{folder}/"
    for images in os.listdir(folder_dir):
        if (images.endswith(".jpeg")):
            full_path = folder_dir + images
            print(full_path)
            img = cv2.imread(full_path)

            # Polygon corner points coordinates
            pts = np.array([[300, 150], [700, 150],
                [300, 150], [30, 700],
                [30, 700],[1100,700],
                [1100, 700], [700, 150]], np.int32)
            

            ## (1) Crop the bounding rect
            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            croped = img[y:y+h, x:x+w].copy()

            ## (2) make mask
            pts = pts - pts.min(axis=0)
            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            ## (3) do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)

            ## (4) add the white background
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst

            #cv2.imwrite("croped.png", croped)
            #cv2.imwrite("mask.png", mask)
            #cv2.imwrite("dst.png", dst)

            filename = images.split('.')[0]
            cv2.imwrite(f"processed/{folder}/{filename}.png", dst2)