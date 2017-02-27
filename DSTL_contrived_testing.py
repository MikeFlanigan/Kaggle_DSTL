import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, sys

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Reshape, Merge
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.constraints import maxnorm
import keras

# fix random seed for reproducible testing
##seed = 7
##np.random.seed(seed)

black = (0,0,0)
white = (255,255,255)
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)

im_w = 40
im_h = 40

blank_target = np.zeros((im_h,im_w,3))
blank_target[:,:,:]=255

def rpt(img, X):
    if X == 0:
        pt = np.random.randint(0,img.shape[0])
    elif X == 1:
        pt = np.random.randint(0,img.shape[1])
    return pt
    

def add_target(img):
##    pts = np.array([[rpt(img,0),rpt(img,1)],[rpt(img,0),rpt(img,1)],[rpt(img,0),rpt(img,1)],[rpt(img,0),rpt(img,1)]])
##    pts = np.array([[0,0],[10,0],[10,10],[0,10]]) # hard coding an actual shape
##    cv2.fillPoly(img, [pts],green)
##    cv2.rectangle(img,(3,3),(20,20),green,-1)
    cv2.rectangle(img,(rpt(img,0),rpt(img,1)),(rpt(img,0),rpt(img,1)),green,-1)

cv2.namedWindow('win',cv2.WINDOW_NORMAL)
cv2.resizeWindow('win',50,50)
    
add_target(blank_target)
##cv2.imshow('win',blank_target)
##cv2.waitKey(0)

X = np.zeros((im_w*im_h,im_h,im_w,3))
X[0:X.shape[0]]=blank_target

# horribly inefficient way of creating pixel of interest input data
X2 = np.zeros((im_w*im_h,2))
ii = 0
n=0
for i in np.arange(im_w*im_h):
    if i > 0 and np.mod(i,im_h)==0:
        ii +=1
        n=0
    X2[i,:]= [ii,n]
    n+=1

lower = (0,0,0)
upper = (0,255,0)
mask = cv2.inRange(blank_target, lower, upper)
cv2.imshow('win',mask)
cv2.waitKey(100)

mask[mask==255]=1
##Y = mask.flatten()
Y = np.reshape(mask,(im_w*im_h,))

# save files for further testing
cv2.imwrite('X1b_image.png',X[0,:,:,:])
np.save('X2b_dat.npy',X2)
np.save('Yb.npy',Y)

sys.exit()
print('X shape:',X.shape,'X2 shape',X2.shape,'Y shape',Y.shape)

####### Machine learning #######

# scale & normalize inputs
for i in np.arange(X.shape[3]):
    mu = X[:,:,:,i].mean() # average value for each feature
    sigma = X[:,:,:,i].std() # standard deviation for each feature
    if sigma != 0: X[:,:,:,i] = (X[:,:,:,i]-mu)/sigma
    else: X[:,:,:,i] = (X[:,:,:,i]-mu)
    print('IMAGE INPUT Max val in feature',i,X[:,:,:,i].max(), 'Min val in feature',i,X[:,:,:,i].min())
for i in np.arange(X2.shape[1]):
    mu = X[:,i].mean() # average value for each feature
    sigma = X[:,i].std() # standard deviation for each feature
    if sigma != 0: X[:,i] = (X[:,i]-mu)/sigma
    else: X[:,i] = (X[:,i]-mu)
    print('PIXEL INPUT Max val in feature',i,X[:,i].max(), 'Min val in feature',i,X[:,i].min())

### shuffling data
##indx = np.arange(X.shape[0]) 
##np.random.shuffle(indx)
##X = X[indx]
##X2 = X2[indx]
##Y = Y[indx]


# Splitting data into train, validation, and test sets
##nintyP = 
eightyP = int(np.floor(X.shape[0]*.8))
sixtyP = int(np.floor(X.shape[0]*.6))
twentyP = int(np.floor(X.shape[0]*.2))
##fiveP = 



pos_vals = np.where(Y==1)[0]
neg_vals = np.where(Y==0)[0]

if len(pos_vals)<len(neg_vals): smaller_class = pos_vals
else: smaller_class = neg_vals

pos_list = []
indx = np.arange(len(pos_vals))
np.random.shuffle(indx)
for i in np.arange(int(len(smaller_class)*3/4)):
    pos_list.append(pos_vals[indx[i]])

neg_list = []
indx = np.arange(len(neg_vals))
np.random.shuffle(indx)
for i in np.arange(int(len(smaller_class)*3/4)):
    neg_list.append(neg_vals[indx[i]])
    

##pos_list = [55, 64, 65, 73, 74, 75, 84, 85, 86]
##neg_list = [0, 10, 19, 34, 38, 50, 61, 77, 88, 91]
X_train = X[pos_list+neg_list,:]
X2_train = X2[pos_list+neg_list,:]
Y_train = np.resize(Y[pos_list+neg_list],(len(pos_list)+len(neg_list),1))

# re-shuffling training data
##indx = np.arange(X_train.shape[0]) 
##np.random.shuffle(indx)
##X_train = X_train[indx]
##X2_train = X2_train[indx]
##Y_train = Y_train[indx]

X_val = X
X2_val = X2
Y_val = np.resize(Y,(Y.shape[0],1))

X_test = X
X2_test = X2
Y_test = np.resize(Y,(Y.shape[0],1))


##X_train = X[0:sixtyP,:]
##X2_train = X2[0:sixtyP,:]
##Y_train = np.resize(Y[0:sixtyP],(sixtyP,1))
##
##X_val = X[sixtyP:sixtyP+twentyP,:]
##X2_val = X2[sixtyP:sixtyP+twentyP,:]
##Y_val = np.resize(Y[sixtyP:eightyP],(twentyP,1))
##
##X_test = X[eightyP:X.shape[0],:]
##X2_test = X2[eightyP:X.shape[0],:]
##Y_test = np.resize(Y[eightyP:X.shape[0]],(X.shape[0]-eightyP,1))

print('Training set shape:',X_train.shape,X2_train.shape,Y_train.shape)
print('Testing set shape:',X_test.shape,X2_test.shape,Y_test.shape)
print('Validation set shape:',X_val.shape,X2_val.shape,Y_val.shape)
##sys.exit()
# make a custom callback since regular tends to prints bloat notes
class cust_callback(keras.callbacks.Callback):
        def __init__(self):
                self.train_loss = []
                self.val_loss = []
        def on_epoch_end(self,epoch,logs={}):
                print('epoch:',epoch,' loss:',logs.get('loss'),' validation loss:',logs.get('val_loss'))
                self.val_loss.append(logs.get('val_loss'))
                self.train_loss.append(logs.get('loss'))
                return
        def on_batch_end(self,batch,logs={}):
                return

history = cust_callback()

# define model
img_model = Sequential()
img_model.add(Convolution2D(8, 3, 3, input_shape=(im_h,im_w, 3)))
##img_model.add(MaxPooling2D(pool_size=(2, 2)))
img_model.add(Activation('relu'))
img_model.add(Convolution2D(8, 3, 3))
img_model.add(Activation('relu'))
img_model.add(Flatten())
img_model.add(Dense(im_w*im_h,init='uniform'))
img_model.add(Activation('relu'))

pixel_model = Sequential()
pixel_model.add(Dense(2,input_dim=2))

merged = Merge([img_model, pixel_model], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(10, activation='relu'))
final_model.add(Dense(1))
final_model.add(Activation('sigmoid'))

print("compiling...")
final_model.compile(optimizer='sgd', loss='binary_crossentropy')

try:
        print("fitting...")
        final_model.fit(
                [X_train,X2_train],
                Y_train,
                validation_data = ([X_val,X2_val],Y_val),
                batch_size=8,
                nb_epoch=50, ## begins overfitting around epoch 600, should implement check points and also take some screen shots!
                verbose = 0,
                callbacks=[history])
except KeyboardInterrupt:
        pass


def cus_eval(dec_thresh,predictions,labels):
        true_pos = len(np.where((predictions[:,0] > dec_thresh) & (labels[:,0] == 1))[0])
        false_pos = len(np.where((predictions[:,0] > dec_thresh) & (labels[:,0] == 0))[0])
        true_neg = len(np.where((predictions[:,0] < dec_thresh) & (labels[:,0] == 0))[0])
        false_neg = len(np.where((predictions[:,0] < dec_thresh) & (labels[:,0] == 1))[0])

        if (true_pos+false_pos)>0: precision = true_pos/(true_pos+false_pos)
        else: precision = 0
        if (true_pos+false_neg) > 0: recall = true_pos/(true_pos+false_neg)
        else: recall = 0
        if (precision+recall)>0: F1score = 2*precision*recall/(precision+recall)
        else: F1score = 0
        return F1score,precision,recall,true_pos,false_pos,true_neg,false_neg

fig_num = 1
plt.figure(fig_num)
fig_num+=1
##axes = plt.gca()
plt.ion()
##axes.set_ylim([0,0.15])
plt.plot(history.train_loss, 'b',label='train loss')
plt.plot(history.val_loss, 'r',label='val loss')
plt.legend()
plt.show()

print('training class dist stats',sum(Y_train),' pos', len(Y_train)-sum(Y_train),'neg')
# --- output evaluation results -----------
dec_thresh = 0.5
print('TRAINING SET RESULTS:')
predictions = final_model.predict([X_train,X2_train]) # make predicitions
F1score,precision,recall,true_pos,false_pos,true_neg,false_neg = cus_eval(dec_thresh,predictions,Y_train)
print('True Pos:',true_pos,'False Pos:',false_pos,'True Neg:',true_neg,'False Neg:',false_neg)
print('F1score:',F1score)
print("accuracy: ",(true_pos+true_neg)/predictions.shape[0])


print('VALIDATION SET RESULTS:')
predictions = final_model.predict([X_val,X2_val]) # make predicitions
F1score,precision,recall,true_pos,false_pos,true_neg,false_neg = cus_eval(dec_thresh,predictions,Y_val) 
print('True Pos:',true_pos,'False Pos:',false_pos,'True Neg:',true_neg,'False Neg:',false_neg)
print('F1score:',F1score)
print("accuracy: ",(true_pos+true_neg)/predictions.shape[0])


print('TEST SET RESULTS:')
predictions = final_model.predict([X_test,X2_test]) # make predicitions
F1score,precision,recall,true_pos,false_pos,true_neg,false_neg = cus_eval(dec_thresh,predictions,Y_test) 
print('True Pos:',true_pos,'False Pos:',false_pos,'True Neg:',true_neg,'False Neg:',false_neg)
print('F1score:',F1score)
print("accuracy: ",(true_pos+true_neg)/predictions.shape[0])


cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)

#thresholding predictions
##predictions[predictions>.5]=1
##predictions[predictions<=.5]=0

pred_img = np.reshape(predictions,(im_h,im_w))
plt.figure(fig_num)
fig_num+=1
plt.title('test predictions')
plt.imshow(pred_img)

t_pred_img = pred_img
t_pred_img[t_pred_img>.5]=1
t_pred_img[t_pred_img<=.5]=0
plt.figure(fig_num)
fig_num+=1
plt.title('thresholded predictions')
plt.imshow(t_pred_img)

true_img = np.reshape(Y_test,(im_h,im_w))
plt.figure(fig_num)
fig_num+=1
plt.title('ground truth')
plt.imshow(true_img,cmap="hot")

##pred_img = pred_img.transpose(1,0)
##plt.figure(fig_num)
##fig_num+=1
##plt.title('transposed predictions')
##plt.imshow(pred_img,cmap="hot")


