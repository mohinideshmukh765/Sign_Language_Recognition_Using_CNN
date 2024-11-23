# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()
# First convolution layer and pooling
classifier.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(sz, sz, 1)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolution layer and pooling
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth convolution layer and pooling
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Fifth convolution layer and pooling
classifier.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Sixth convolution layer without reducing dimensions further
classifier.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Fully connected layers with Dropout for regularization
classifier.add(Dense(units=256, activation='relu', kernel_regularizer=l2(0.01)))
classifier.add(Dropout(0.5))

# New fully connected layer with 512 units
classifier.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.01)))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
classifier.add(Dropout(0.5))
# Output layer (adjust units if you have more or fewer classes)
classifier.add(Dense(units=27, activation='softmax'))
#from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# Optimizer with a lower initial learning rate
#classifier.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
from keras.metrics import AUC, Precision, Recall
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC()])
#classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

# Step 2 - Preparing the train/test data and training the model
classifier.summary()
# Code copied from - https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( 
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 1.5],
    channel_shift_range=20.0  # Add this for simulating light conditions

)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory("C:/VSCode/project/data/train",
                                                 target_size=(sz, sz),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='categorical',
                                                classes=['_BLANK','A','B','C','D','E,','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']) 


test_set = test_datagen.flow_from_directory("C:/VSCode/project/data/test",
                                            target_size=(sz , sz),
                                            batch_size=32,
                                            color_mode='grayscale',
                                            class_mode='categorical',
                                           classes=['_BLANK','A','B','C','D','E,','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']) 


#steps_per_epoch = len(training_set)
#validation_steps = len(test_set)
# Increase the number of epochs
EPOCHS = 10
BATCH_SIZE = 32

from keras.callbacks import EarlyStopping

# Set up the ModelCheckpoint callback
checkpoint = ModelCheckpoint('best_weights.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=EPOCHS,
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[reduce_lr, early_stopping, checkpoint]
    
)
# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model-bw.h5')
print('Weights saved')