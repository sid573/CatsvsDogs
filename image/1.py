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



fn = sys.argv[1]
model = load_model('my_model.h5')
#model.load_weights('my_model_weights.h5')


#yprint("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# summarize history for accuracy
# summarize history for loss

test_image=image.load_img(fn,target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
print(result)