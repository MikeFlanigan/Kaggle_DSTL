import pandas as pd
import numpy as np 
import cv2
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import matplotlib.pyplot as plt




##
##DF = pd.read_csv('train_wkt_v4.csv')
##GS = pd.read_csv('grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
##SB = pd.read_csv('sample_submission.csv')





# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##from subprocess import check_output
##print(check_output(["ls", "../input"]).decode("utf8"))




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

##class data_class:
##    def __init__(self,label):
##        self.name = class_labels[str(label)]
##        self.


# read the training data 
df = pd.read_csv('train_wkt_v4.csv')
# grid size 
gs = pd.read_csv('grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

train_images = np.unique(df.ImageId[0:250])

class_totals = []
class_img_dists = []
for d_class in np.arange(1,11):
    print('Class:',d_class,' :',class_labels[str(d_class)])
    perc_class_in_img = 0
    class_img_dists.append(d_class)
    class_img_dists[d_class-1]=[]
    for img_name in train_images:

        mask = generate_mask_for_image_and_class((800,800),img_name,d_class,gs,df)
        perc_class_in_img += (mask.sum()/(mask.shape[0]*mask.shape[1]))
        class_img_dists[d_class-1].append(mask.sum()/(mask.shape[0]*mask.shape[1])*100)
##        print('percent in image:',mask.sum()/(mask.shape[0]*mask.shape[1]))
    class_totals.append(perc_class_in_img)
    
objects = (class_labels["1"],class_labels["2"],class_labels["3"],class_labels["4"],class_labels["5"],
           class_labels["6"],class_labels["7"],class_labels["8"],class_labels["9"],class_labels["10"],)
y_pos = np.arange(1,11)
performance = class_totals
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.ylim([0,np.max(class_totals)+.25*np.max(class_totals)])
plt.xticks(y_pos, objects,rotation='vertical')
plt.ylabel('class % area over training set')
plt.title('Training data class distribution')
plt.ion()
plt.show()

if True: # show distribution for each class in each image
    for i in np.arange(1,11):
        plt.figure(i+1)
        plt.bar(np.arange(len(class_img_dists[i-1])), class_img_dists[i-1], align='center', alpha=0.5)
        y_pos = np.arange(25)
        plt.xticks(y_pos, train_images,rotation='vertical')
        plt.ylim([0,np.max(class_img_dists[i-1])+.25*np.max(class_img_dists[i-1])])
        plt.ylabel('class distribution over training set images')
        plt.title(class_labels[str(i)])
        plt.show()


    
    

    # saving images
##    cv2.imwrite("class_"+class_labels[str(d_class)]+"_mask.png",mask*255)
    # displaying images
##    cv2.imshow('win',mask*255)
##    while True:
##        key = cv2.waitKey(1) & 0xFF
##        if key == ord(' '): break
        
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
