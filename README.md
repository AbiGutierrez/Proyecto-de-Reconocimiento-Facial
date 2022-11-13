# Red neuronal artificial de reconocimiento facial 

_Se presenta una RNA de reconocimiento facial, el objetivo de la misma es que la red sea capaz de indentificar al autor de la red._

## Estrategía 🚀

**Descripción de la base de datos:**
El conjunto de datos de atributos de CelebFaces (CelebA) es un conjunto de datos de atributos faciales a gran escala con más de 200 000 imágenes de celebridades, cada una con 40 anotaciones de atributos.
- 10,177 número de identidades
- 202,599 número de imágenes de rostros
- 40 anotaciones de atributos binarios por imagen.

Se procesan los datos de las base descargada (CelebA), para el preprocesamiento de los datos primero se eliminaron los dobles espacios en el dataframe debido a que 
estaban ocasionando errores en el momento de leer los datos. 

```
with open('list_attr_celeba.txt', 'r') as f:

    print("skipping : " + f.readline())    #salatarnos la primera fila
    
    print("skipping headers :" + f.readline()) #saltarnos la segunda fila la de los encabezados
        
    with open('list_attr_celeba_prepared.txt' , 'w') as newf:
            
    for line in f:
                    
    new_line = ' '.join(line.split())  #separa las palabras de una frase e ilimina los espacios 
                                
    newf.write(new_line)
                                            
    newf.write('\n')
```
    


Asi el dataFrame queda separado por comas, se procede juntar cada imagen con sus respectivas etiquetas de caracteristicas. 
```
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
model.add(Conv2D(32, (3, 3), input_shape=(192, 192, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
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


### 🔧



## Ejecutando las pruebas ⚙️

### Analice las pruebas end-to-end 🔩



