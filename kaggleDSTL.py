import os, sys, re
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import numpy as np

six_band = True
six_band = False
##cv2.namedWindow('test',cv2.WINDOW_NORMAL)
##cv2.resizeWindow('test',int(1920/4),int(1080/4))

    
sx_bd_imgs = os.listdir('./sixteen_band/')
sx_bd_imgs = sorted(sx_bd_imgs, key=lambda x: (int(re.sub('\D','',x)),x))
tre_bd_imgs = os.listdir('./three_band/')
tre_bd_imgs = sorted(tre_bd_imgs, key=lambda x: (int(re.sub('\D','',x)),x))
trn_geojson_files = os.listdir('./train_geojson_v3')

a_file = './three_band/6010_3_0.tif'

stop = False
img_num = 1
if not six_band:
    start_img = np.random.randint(0,len(tre_bd_imgs)-1)
    for img in tre_bd_imgs:
        if img_num < start_img:
            img_num +=1
            continue
        else: pass
        print('showing ',img,' img',img_num,' of ',len(tre_bd_imgs))
        img_num +=1
        P = tiff.imread('./three_band/'+img)
        print('shape',P.shape,'min',P.min(),'max',P.max(),'mean',P.mean())
        tiff.imshow(P)
        plt.title(img)
        plt.show()
        while plt.fignum_exists(1): pass 

else:
    start_img = np.random.randint(0,len(sx_bd_imgs)-1)
    for img in sx_bd_imgs:
        if img_num < start_img:
            img_num +=1
            continue
        else: pass
        print('showing ',img,' img',img_num,' of ',len(sx_bd_imgs))
        img_num +=1
        P = tiff.imread('./sixteen_band/'+img)
        print('shape',P.shape,'min',P.min(),'max',P.max(),'mean',P.mean())
        tiff.imshow(P)
        plt.title(img)
##        plt.title(img, y = 1)
        plt.show()
        while plt.fignum_exists(1): pass 
            



