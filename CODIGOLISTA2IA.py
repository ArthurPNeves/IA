# %%
import tensorflow as tf
import keras as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# %%
# Inicializando a Rede Neural Convolucional

classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compilando a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# %%
# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory('./dataset_personagens/dataset_personagens/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('./dataset_personagens/dataset_personagens/test_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

# %%
# Executando o treinamento (esse processo pode levar bastante tempo, dependendo do seu computador)
classifier.fit(training_set,
                         steps_per_epoch = 3,
                         epochs = 200,
                         validation_data = validation_set,
                         validation_steps = 3)

# %%
# Primeira Imagem
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

test_image = image.load_img(
    'dataset_personagens/dataset_personagens/training_set/bart/bart102.bmp', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

img = mpimg.imread('dataset_personagens/dataset_personagens/training_set/bart/bart102.bmp')
plt.imshow(img)
plt.axis('off')  # Desativa os eixos
plt.title(f"Prediction: {prediction}")
plt.show()

# dummy = Image(filename='dataset_personagens/training_set/bart/bart55.bmp')

# %%
# Segunda Imagem
test_image = image.load_img('dataset_personagens/dataset_personagens/training_set/homer/homer102.bmp', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

img = mpimg.imread('dataset_personagens/dataset_personagens/training_set/homer/homer102.bmp')
plt.imshow(img)
plt.axis('off')  # Desativa os eixos
plt.title(f"Prediction: {prediction}")
plt.show()
#Image(filename='dataset_personagens/training_set/bart/bart55.bmp')


