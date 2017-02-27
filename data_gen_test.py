from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Reshape, Merge
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.constraints import maxnorm
import keras
import numpy as np


train_data_dir = './dataX1/train'
validation_data_dir = './dataX1/validation'
test_data_dir = './dataX1/test'


train_datagen = ImageDataGenerator(
        rescale=1./255, # will change based on different channel maxes
        horizontal_flip=True,
        vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(40, 40),
        batch_size=1,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(40, 40),
        batch_size=1,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(40, 40),
        batch_size=1,
        class_mode='binary')

X2 = np.load('X2_dat.npy')
X2 = np.resize(X2,(X2.shape[0],X2.shape[1],1,1))
Y2 = np.load('Y.npy')

train_gen2 = ImageDataGenerator()
def format_gen_outputs(gen1,gen2):
    x1 = gen1[0]
    x2 = gen2[0]
    y1 = gen2[1]
    return [x1, x2], y1
train_combo_gen = map(format_gen_outputs, train_generator, train_gen2.flow(X2, Y2, batch_size=1))


X2val = np.load('X2a_dat.npy')
X2val = np.resize(X2val,(X2val.shape[0],X2val.shape[1],1,1))
Y2val = np.load('Ya.npy')

val_gen2 = ImageDataGenerator()
val_combo_gen = map(format_gen_outputs, validation_generator, val_gen2.flow(X2val, Y2val, batch_size=1))

# define model
img_model = Sequential()
img_model.add(Convolution2D(8, 3, 3, input_shape=(40,40, 3)))
##img_model.add(MaxPooling2D(pool_size=(2, 2)))
img_model.add(Activation('relu'))
img_model.add(Convolution2D(8, 3, 3))
img_model.add(Activation('relu'))
img_model.add(Flatten())
img_model.add(Dense(40*40,init='uniform'))
img_model.add(Activation('relu'))

### should not be using a convolution here.... but will try
pixel_model = Sequential()
pixel_model.add(Convolution2D(2, 1, 1, input_shape=(2,1,1)))
pixel_model.add(Activation('relu'))
pixel_model.add(Flatten())
##pixel_model.add(Dense(2,input_dim=2))

merged = Merge([img_model, pixel_model], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(10, activation='relu'))
final_model.add(Dense(1))
final_model.add(Activation('sigmoid'))

print("compiling...")
final_model.compile(optimizer='sgd', loss='binary_crossentropy')

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

final_model.fit_generator(
        train_combo_gen,
        samples_per_epoch=16,
        nb_epoch=2,
        validation_data=val_combo_gen,
        nb_val_samples=8,
        verbose = 0,
        callbacks=[history])


##predictions = final_model.predict_generator(combo_gen, val_samples=16, max_q_size=10, nb_worker=1, pickle_safe=False)
