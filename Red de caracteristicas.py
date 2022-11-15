# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:21:45 2022

@author: abigu
"""

import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
#import pathlib
import matplotlib.pyplot as plt
import os
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
#from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split

#np.set_printoptions(precision=4)

#eliminar el dodle espacio 
'''with open('list_attr_celeba.txt', 'r') as f:
    print("skipping : " + f.readline())    #salatarnos la primera fila
    print("skipping headers :" + f.readline()) #saltarnos la segunda fila la de los encabezados
    with open('list_attr_celeba_prepared.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())  #separa las palabras de una frase e ilimina los espacios 
            newf.write(new_line)
            newf.write('\n')  #simbolo de la nueva linea'''

#carga de base de datos

df = pd.read_csv('list_attr_celeba_prepared.txt', sep=' ', header=None)  #dataframe
df = df.replace({-1:0})
#print(df)
#print(df.head)
#print("--------")
#print(df[0].head())
#print(df.iloc[:,1:].head)
#print("---------------")

#se pasan los datos a un tensor de tf y se juntan
files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, attributes))
#print(data) 


#procesar imagenes
path_to_images = 'img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

images_labeled= data.map(process_file)
#image, attributes = data.map(process_file)  #map evalua una funcion en una lista
#image, attributes = data.map(process_file)
#train= data.map(process_file)




#x, x_test, y, y_test = train_test_split((image, attributes), test_size=0.2, random_state=1)

'''print(images_labeled) 
for image, attri in images_labeled.take(1):
    plt.imshow(image)
    plt.show()
    print(attri)'''
    

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
fun_loss = tf.keras.losses.MeanSquaredError()
num_class = 40 #cuantas clases
epochs =15#cuantas eces entrenar. En cada epoch hace una mejora en los parametros
batch_size = 50 #batch para hacer cada entrenamiento. Lee 50 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria
#batch_test = images_labeled.batch(batch_size)  
images_labeled = images_labeled.batch(batch_size)  
#image, attributes = (image, attributes).batch(batch_size)
#train = train.shuffle(buffer_size=10 * batch_size)
#batch_train = train.batch(80)
'''base_model = Xception(input_shape=(192, 192, 3), include_top=False) # Average pooling reduces output dimensions
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(40, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)'''

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(192, 192, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(40))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss=fun_loss, 
              optimizer = opt,
              metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', 
 #             optimizer = 'adadelta',
  #            metrics=['accuracy'])
model_history = model.fit(images_labeled, epochs=epochs,
                          validation_data=images_labeled,
                                    batch_size=batch_size)
#model_history = model.fit(images, attribures, batch_size=images_labeled, epochs=epochs,verbose=1,
                   #validation_data=images, attributes)
#score = model.evaluate(image_labeled, verbose=0)
#print(score)


def plotTraining(hist, epochs, typeData):
    
    if typeData=="loss":
        plt.figure(1,figsize=(10,5))
        yc=hist.history['loss']
        xc=range(epochs)
        plt.ylabel('Loss', fontsize=24)
        plt.plot(xc,yc,'-r',label='Loss Training')
    if typeData=="accuracy":
        plt.figure(2,figsize=(10,5))
        yc=hist.history['accuracy']
        for i in range(0, len(yc)):
            yc[i]=100*yc[i]
        xc=range(epochs)
        plt.ylabel('Accuracy (%)', fontsize=24)
        plt.plot(xc,yc,'-r',label='Accuracy Training')
    if typeData=="val_loss":
        plt.figure(1,figsize=(10,5))
        yc=hist.history['val_loss']
        xc=range(epochs)
        plt.ylabel('Loss', fontsize=24)
        plt.plot(xc,yc,'--b',label='Loss Validate')
    if typeData=="val_accuracy":
        plt.figure(2,figsize=(10,5))
        yc=hist.history['val_accuracy']
        for i in range(0, len(yc)):
            yc[i]=100*yc[i]
        xc=range(epochs)
        plt.ylabel('Accuracy (%)', fontsize=24)
        plt.plot(xc,yc,'--b',label='Training Validate')
        

    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    plt.rc('legend', fontsize=18) 
    plt.legend()
    plt.xlabel('Number of Epochs',fontsize=24)
    plt.grid(True)
    
plotTraining(model_history,epochs,"loss")
plotTraining(model_history,epochs,"accuracy")
plotTraining(model_history,epochs,"val_loss")
plotTraining(model_history,epochs,"val_accuracy")
