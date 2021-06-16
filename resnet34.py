#coding=utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D
from tensorflow.keras.layers import add,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
seed = 7
np.random.seed(seed)



DATASET_PATH  = '../dataset/Resnet_data/argument'

IMAGE_SIZE = (300, 300)


NUM_CLASSES = 2


BATCH_SIZE = 16


FREEZE_LAYERS = 2

NUM_EPOCHS = 20

WEIGHTS_FINAL = 'model-resnet34-20t_sgd.h5'


train_datagen = ImageDataGenerator()
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)


for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))


def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x
 
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x

inpt = Input(shape=(300,300,3))
x = ZeroPadding2D((3,3))(inpt)
x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

#(56,56,64)
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
#(28,28,128)
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
#(14,14,256)
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#(7,7,512)
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
x = AveragePooling2D(pool_size=(7,7))(x)
x = Flatten()(x)
x = Dense(NUM_CLASSES,activation='softmax')(x)

model = Model(inputs=inpt,outputs=x)
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()

model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)


prediction=model.predict_generator(valid_batches,verbose=1)

predict_label=np.argmax(prediction,axis=1)
true_label=valid_batches.classes



print(pd.crosstab(true_label,predict_label,rownames=['label'],colnames=['predict']))


model.save(WEIGHTS_FINAL)


