import os, sys, re
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import matplotlib.pyplot as plt


processed_list = []

one_image = '6060_2_3' # positive image # negative image 6010_4_2

sixteen_dir = './sixteen_band/'
three_dir = './three_band/'

RGB = tiff.imread(three_dir+one_image+'.tif')
A = tiff.imread(sixteen_dir+one_image+'_A.tif')
M = tiff.imread(sixteen_dir+one_image+'_M.tif')
P = tiff.imread(sixteen_dir+one_image+'_P.tif')

RGB=RGB.transpose(1,2,0).astype('int16')
A=A.transpose(1,2,0)
M=M.transpose(1,2,0)

A = cv2.resize(A,(RGB.shape[1],RGB.shape[0])).astype('int16')
M = cv2.resize(M,(RGB.shape[1],RGB.shape[0])).astype('int16')
P = cv2.resize(P,(RGB.shape[1],RGB.shape[0])).astype('int16')


twenty_chan_img = np.concatenate((RGB,A,M,np.resize(P,(P.shape[0],P.shape[1],1))),2)

# scale normalize X
for chan in np.arange(twenty_chan_img.shape[2]):
    mu = np.mean(twenty_chan_img[:,:,chan])
    sigma = np.std(twenty_chan_img[:,:,chan])
    if sigma > 0: twenty_chan_img[:,:,chan] = (twenty_chan_img[:,:,chan]-mu)/sigma
    else: twenty_chan_img[:,:,chan] = (twenty_chan_img[:,:,chan]-mu)
    print('Max val in feature',chan,twenty_chan_img[:,:,chan].max())



### get and create a mask

def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d


def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax,Ymax = xymax
    H,W = img_size
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/Xmax
    yf = H1/Ymax
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):
    img_mask = np.zeros(raster_img_size,np.uint8)
    if contours is None:
        return img_mask
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list,class_value)
    cv2.fillPoly(img_mask,interior_list,0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,1)
    return mask


class_labels = {"1":"Buildings","2":"Misc_Structs",
            "3":"Road","4":"Track","5":"Trees",
            "6":"Crops","7":"Waterway",
            "8":"Still_Water","9":"Big_Vehicle",
            "10":"Small_Vehicle"}

# read the training data 
dstl_labels = pd.read_csv('train_wkt_v4.csv')
# grid size 
grid_size = pd.read_csv('grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

mask = generate_mask_for_image_and_class((RGB.shape[0],RGB.shape[1]),one_image,6,grid_size,dstl_labels) # mask


# create a sliding window and save each image patch as either a positive or a negative
win_rows, win_cols = 100,100

# array of indices of all window centers
# horribly inefficient way
tot_indices = (mask.shape[0]-win_rows)*(mask.shape[1]-win_cols)
win_c_inds = np.zeros((tot_indices,2))
row = 0
col = 0
j = 0
win_c_inds = np.load('win_cs.npy') # only if have an accurate saved one
##for ii in np.arange(tot_indices):
##    if col > 0 and np.mod(col,mask.shape[1])==0:
##        row +=1
##        col=0
##        if np.mod(row,100)==0: print(row)
##    if row > win_rows/2 and row < mask.shape[0]-win_rows/2 and col > win_cols/2 and col < mask.shape[1] - win_cols/2:
##        win_c_inds[j,:] = np.array([row, col])
##        j+=1
##    col+=1
    
# shuffle the array
##print('shuffling index array...')
##np.random.shuffle(win_c_inds)

# training and positives 
# access the array and save on positive or negative
pos_dir = './data/train/positive/'
neg_dir = './data/train/negative/'
print('creating image patches and saving...')
ptc = 0 # postiive training count
ntc = 0 # negative...
while ptc < mask.sum()*.8:
    im_patch = twenty_chan_img[win_c_inds[ptc,0]-win_rows/2:win_c_inds[ptc,0]+win_rows/2,win_c_inds[ptc,1]-win_cols/2:win_c_inds[ptc,1]+win_cols/2,:]
    if mask[int(win_c_inds[ptc,0]),int(win_c_inds[ptc,1])]==1:
        tiff.imsave(pos_dir+one_image+'_'+str(ptc)+'.tif',im_patch)
##        cv2.imwrite(pos_dir+one_image+'_'+str(ptc)+'.jpg',im_patch)
        ptc+=1
    elif mask[int(win_c_inds[ptc,0]),int(win_c_inds[ptc,1])]==0:
        tiff.imsave(neg_dir+one_image+'_'+str(ntc)+'.tif',im_patch)
        ntc +=1
    else: print('ERROR MASK VAL',mask[int(win_c_inds[ptc,0]),int(win_c_inds[ptc,1])])
    if np.mod(ptc,100)==0: print('ptc:',ptc)
    if np.mod(ntc,100)==0:print('ntc:',ntc)
    
##for row in np.arange(mask.shape[1]):
##    for col in np.arange(mask.shape[0]):
        

##cv2.imshow('mask',np.resize(mask,(1000,1000))*255)
##cv2.waitKey(1)
##cv2.waitKey(1)
##cv2.waitKey(1)
##cv2.waitKey(1)

