
get_ipython().magic('matplotlib inline')

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
img_width, img_height = 224, 224

n_classes = 128
train_data_dir = 'train_chunk1'
validation_data_dir = 'valid'

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

train_generator_bottleneck = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

validation_generator_bottleneck = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

nb_train_samples = len(train_generator_bottleneck.filenames)  
print(nb_train_samples)

model_vgg = applications.VGG16(include_top=False, weights='imagenet')

import math
predict_size_train = int(math.ceil(nb_train_samples / batch_size))  
bottleneck_features_train = model_vgg.predict_generator(train_generator_bottleneck,predict_size_train) 
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

predict_size_valid = int(math.ceil(6309 / batch_size))  
bottleneck_features_validation = model_vgg.predict_generator(validation_generator_bottleneck,predict_size_valid)
np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

num_classes = len(train_generator_bottleneck.class_indices)

train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
train_labels = to_categorical(train_generator_bottleneck.classes, num_classes=num_classes)

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels = to_categorical(validation_generator_bottleneck.classes, num_classes=num_classes)

print(type(train_data))

model_top = Sequential()
model_top.add(Flatten(input_shape=train_data.shape[1:]))
model_top.add(Dense(512, activation='relu'))
model_top.add(Dropout(0.3))
model_top.add(Dense(n_classes, activation='softmax'))

model_top.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

epochs = 100
train_samples = 164171
validation_samples = 6309

checkpointer = ModelCheckpoint(filepath='bottleneck_features.h5', monitor='val_acc', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=5)

history = model_top.fit(
        train_data,
        train_labels,
        verbose=2,
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=[checkpointer, early_stopping],
        validation_data=(validation_data, validation_labels))


model_top.evaluate(validation_data, validation_labels)

fig, ax = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
fig.savefig('bottleneck_features.svg', format='svg', dpi=1200)

with open('bottleneck_features.json', 'w') as f:
    f.write(model_top.to_json())

