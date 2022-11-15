# Red neuronal artificial de reconocimiento facial 

_Se presenta una RNA de reconocimiento facial, el objetivo de la misma es que la red sea capaz de indentificar al autor de la red._

## Estrategía 🚀

**Descripción de la base de datos:**
El conjunto de datos de atributos de CelebFaces (CelebA) es un conjunto de datos de atributos faciales a gran escala con más de 200 000 imágenes de celebridades, cada una con 40 anotaciones de atributos.
- 10,177 número de identidades
- 202,599 número de imágenes de rostros
- 40 anotaciones de atributos binarios por imagen.

Se cargo el dataframe 
```
with open('list_attr_celeba.txt', 'r') as f:
    print("skipping : " + f.readline())    #salatarnos la primera fila
    print("skipping headers :" + f.readline()) #saltarnos la segunda fila la de los encabezados
    with open('list_attr_celeba_prepared.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())  #separa las palabras de una frase e ilimina los espacios 
            newf.write(new_line)
            newf.write('\n')  #simbolo de la nueva linea
          
#carga de base de datos

df = pd.read_csv('list_attr_celeba_prepared.txt', sep=' ', header=None)  #dataframe
df = df.replace({-1:0})
```

Se procesan los datos de las base descargada (CelebA), para el preprocesamiento, se definio una función y se redimensionaron las imagenes a un tamaño de 192x192, se normalizo la intensidad de los pixeles y se junto cada imagen con su respectiva etiqueta. 

```
#procesar imagenes
path_to_images = 'img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

images_labeled = data.map(process_file) 
``` 

**Red neuronal de características**

Se utilizo una red neuronal convolucional, la base de datos contiene 40 características por lo tanto son 40 clases las que se clasificaron.

**Modelo de la red**
```
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
model.add(Dense(40))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy', 
              optimizer = opt,
              metrics=['accuracy'])
model_history = model.fit(images_labeled, epochs=epochs,
                          validation_data=images_labeled,
                                    batch_size=batch_size)
```
Se planeaba hacer el entrenamiento de la red, una vez entranada la red de características, para proceder a hacer el **Transfer Learning**, congelar las primeras capas de la red ya pre entrenada con sus pesos guardados, una vez congeladas las primeras capas se entrenarían las ultimas capas con una base de datos de fotos de uno mismo, ya que solo se contaban con 45 imagenes de entrenamiento y 18 de testing, se opto por aunmentar sinteticamente la base datos con la class **Image Data Generator**.

Primero se probo generando imagenes en una carpeta, se generaron 5 imagenes por cada 1 original en el caso de los datos de entrenamiento.
```
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

```

Y de la misma manera para los datos de testing, solo que en este caso fueron 
```
new_imageTest_folder = 'new_imageTest'
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
    
print("images generadas", num_images)

```
Las imagenes se generaron, sin embargo guardar imagenes en disco no era lo más eficiente, por lo tanto se opto por generar imagenes duarante el entrenamiento de las ultimas de las capas de la red de caracteristicas.

```
ih = 192
epochs = 30
batch_size =30

train_data_dir = 'C:/Users/abigu/Documents/codigos/optativa redes neuronales/proyecto de reconocimiento facial/fotosAbiTrain'
test_data_dir = 'C:/Users/abigu/Documents/codigos/optativa redes neuronales/proyecto de reconocimiento facial/fotosAbiTest'
#train_data_dir = 'fotosAbiTrain' 
#test_data_dir = 'fotosAbiTest'
nuevas_imagenes = 'C:/Users/abigu/Documents/codigos/optativa redes neuronales/proyecto de reconocimiento facial/nuevas_imagenes'
   
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   horizontal_flip=True, 
                                   vertical_flip=False, 
                                   preprocessing_function=(preprocess_input))

test_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   horizontal_flip=True, 
                                   vertical_flip=False, 
                                   preprocessing_function=(preprocess_input))

train_generator = train_datagen.flow_from_directory(train_data_dir,
    target_size=(iw, ih), color_mode="rgb",
    batch_size=batch_size,
    #save_to_dir='nuevasImagenesTrain', save_format='JPG',
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_data_dir,
    target_size=(iw, ih), color_mode="rgb",
    batch_size=batch_size,
    #save_to_dir='nuevasImagenesTest', save_format='JPG',
    class_mode='binary')
```

## Problemas encontrados 

- Lectura del dataframe
- El Dataframe contenia (-1)
- Problema en el entrenamiento de la red caracteristicas 

**Solución de problemas**

En el preprocesamiento los datos se eliminaron los dobles espacios en el dataframe debido a que estaban ocasionando errores en el momento de leer los datos. 
```
#eliminar el dodle espacio 
with open('list_attr_celeba.txt', 'r') as f:
    print("skipping : " + f.readline())    #salatarnos la primera fila
    print("skipping headers :" + f.readline()) #saltarnos la segunda fila la de los encabezados
    with open('list_attr_celeba_prepared.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())  #separa las palabras de una frase e ilimina los espacios 
            newf.write(new_line)
            newf.write('\n')  #simbolo de la nueva linea
```
Los datos se pasaron a un tensor
```
files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, attributes))
```
De esta manera se solucionaron los problemas para la elaboración del dataframe.

Para la eliminación de los (-1) en dataframe, que tambien se considero que estaba ocasionando errores en el entrenamiento, se ultilizo el siguiente comando de la libreria pandas
```
df = pd.read_csv('list_attr_celeba_prepared.txt', sep=' ', header=None)  #dataframe
df = df.replace({-1:0}) #eliminación de los (-1)
```

Una vez establecido el modelo de la red, se procedería a entrenar pero se encontro el siguiente error 
```
ValueError: Input 0 of layer "sequential_4" is incompatible with the layer: expected shape=(None, 192, 192, 3), found shape=(192, 192, 3)
```
Se soluciono implementando la función batch a nuestros datos de entrenamiento 
```
batch_size = 50
images_labeled = data.map(process_file) 
images_labeled = images_labeled.batch(batch_size)

```

Sin embargo una vez solucionado ese problema, se procedio a entrenar la red pero esta no obtuvo aprendizaje, se obtuvo una función de costo negativa y la red tenia una accuracy muy pequeño y se mantenia en el mismo valor.

```
Epoch 77/80
1/1 [==============================] - 1s 862ms/step - loss: nan - accuracy: 0.1220 - val_loss: nan - val_accuracy: 0.1220
Epoch 78/80
1/1 [==============================] - 1s 925ms/step - loss: nan - accuracy: 0.1220 - val_loss: nan - val_accuracy: 0.1220
Epoch 79/80
1/1 [==============================] - 1s 847ms/step - loss: nan - accuracy: 0.1220 - val_loss: nan - val_accuracy: 0.1220
Epoch 80/80
1/1 [==============================] - 1s 835ms/step - loss: nan - accuracy: 0.1220 - val_loss: nan - val_accuracy: 0.1220
```
![Figure1](https://user-images.githubusercontent.com/91716462/201557723-a690fe92-8874-46a8-9343-e8a77b087f4b.png)

Como se puede ver no hubo aprendizaje, se consideraron varias posibilidades que estuvieran provocando el error y tambien intentaron varias opciones para solucionarlo, que a continuación se describen:
- Que solo estuviera considerandose la cantidad del minibatch para entrenar 
- El mini batch muy pequeño
- Los (-1) del dataframe
- Tomar en una función (images, attributes), como (images_labeled), se separaron y se redefinieron como:
```
image, attributes = data.map(process_file)
```
- Función de costo, se probaron funciones dierentes: categorical_crossentropy, mse
- Optimizador, se utilizaron los siguientes: SGD, adadelta, Adam, RMSprop
- Diseño de la red, se probo una red prediseñada llamada Xception, se obtuvo el mismo resultado

De esta forma se obtuvo un resultado negativo y no se pudo proseguir con la elaboración de la red de reconocimiento facial. Se sigue buscando el error que ocasiona el loss:nan e impletando alguna solución efectiva.

## Ejecutando las pruebas ⚙️

### Analice las pruebas end-to-end 🔩



