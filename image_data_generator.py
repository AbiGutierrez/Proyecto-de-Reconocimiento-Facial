# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:33:11 2022

@author: abigu
"""

import os 
import tensorflow as tf
from tensorflow import keras
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, image, img_to_array
import cv2
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import image
from keras.preprocessing.image import ImageDataGenerator

new_imageA_folder = 'new_imageA'
cantidad_de_imagenes = 5

try:
    os.mkdir(new_imageA_folder)
except:
    print("")
    
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   horizontal_flip=True, 
                                   vertical_flip=False)

data_path =  "C:/Users/abigu/Documents/codigos/optativa redes neuronales/proyecto de reconocimiento facial/fotosAbiTrain/AbiTrain"
data_dir_list = os.listdir(data_path)

width_sahape, height_shape = 192, 192

i=0
num_images=0
for image_file in data_dir_list:
    img_list=os.listdir(data_path)

    img_path = data_path+'/'+image_file
    
    imge=load_img(img_path)
    #imge=load_img(img_path)
    
    #imge=tf.image.resize(tf.keras.utils.img_to_array(imge), (width_sahape, height_shape), 
     #               interpolation = cv2.INTER_AREA)
    imge=tf.image.resize(tf.keras.utils.img_to_array(imge), (width_sahape, height_shape))
    #imge=cv2.resize(image.img_to_array(imge), (width_sahape, height_shape), 
     #               interpolation = cv2.INTER_AREA)
    x=imge/255.
    x=np.expand_dims(x, axis=0)
    t=1
    for output_batch in train_datagen.flow(x, batch_size=1):
        #a=image.img_to_array(output_batch[0])
        a=tf.keras.utils.img_to_array(output_batch[0])
        imagen=output_batch[0,:,:]*255.
        imgfinal = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_imageA_folder+"/%i%i.jpg"%(i,t), imgfinal)
        t+=1
        
        num_images+=1
        if t>cantidad_de_imagenes:
            break
    i+=1
    
print("images generadas", num_images)


#--------------------------------------------------------------------

'''new_imageTest_folder = 'new_imageTest'
cantidad_de_imagenes = 2

try:
    os.mkdir(new_imageTest_folder)
except:
    print("")
    
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   horizontal_flip=True, 
                                   vertical_flip=False)

data_path =  "C:/Users/abigu/Documents/codigos/optativa redes neuronales/proyecto de reconocimiento facial/fotosAbiTest"
data_dir_list = os.listdir(data_path)

width_sahape, height_shape = 192, 192

i=0
num_images=0
for image_file in data_dir_list:
    img_list=os.listdir(data_path)

    img_path = data_path+'/'+image_file
    
    imge=load_img(img_path)
    #imge=load_img(img_path)
    
    #imge=tf.image.resize(tf.keras.utils.img_to_array(imge), (width_sahape, height_shape), 
     #               interpolation = cv2.INTER_AREA)
    imge=tf.image.resize(tf.keras.utils.img_to_array(imge), (width_sahape, height_shape))
    #imge=cv2.resize(image.img_to_array(imge), (width_sahape, height_shape), 
     #               interpolation = cv2.INTER_AREA)
    x=imge/255.
    x=np.expand_dims(x, axis=0)
    t=1
    for output_batch in train_datagen.flow(x, batch_size=1):
        #a=image.img_to_array(output_batch[0])
        a=tf.keras.utils.img_to_array(output_batch[0])
        imagen=output_batch[0,:,:]*255.
        imgfinal = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_imageTest_folder+"/%i%i.jpg"%(i,t), imgfinal)
        t+=1
        
        num_images+=1
        if t>cantidad_de_imagenes:
            break
    i+=1
    
print("images generadas", num_images)'''