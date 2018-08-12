from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import sys
import keras
import numpy as np
import h5py
from sklearn.cross_validation import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
#model = load_model('my_model_weights.h5')
batch_size = 16

#fn = sys.argv[1]
model=Sequential()
model.add(Conv2D(filters=128,kernel_size=(5,5),strides=(1,1),padding='valid',input_shape=(64,64,3),activation='relu',use_bias=True))
model.add(MaxPooling2D(pool_size=(3,3),strides=None,padding='valid'))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',use_bias=True))
model.add(MaxPooling2D(pool_size=(3,3),strides=None,padding='valid'))
model.add(Flatten())
model.add(Dense(128,activation='relu',use_bias=True))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('dataset/train',target_size=(64,64),class_mode='binary')
test_set=train_datagen.flow_from_directory('dataset/test',target_size=(64,64),class_mode='binary')
Model = model.fit_generator(training_set,epochs=50,steps_per_epoch = 2000 // batch_size,validation_data=test_set,validation_steps=800 // batch_size)
#yprint("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('my_model.h5')
model.save_weights('my_model_weights.h5')

model.summary()

print(Model.history.keys())

# summarize history for accuracy
plt.plot(Model.history['acc'])
plt.plot(Model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(Model.history['loss'])
plt.plot(Model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

