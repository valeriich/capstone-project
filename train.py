
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.lite as tflite



""" Data reading and preparation"""

root = './tomatoes'

# function to make dataframe with two columns - name of the image file, its class
def make_dataframe(path):
    x_col = list()
    y_col = list()

    for subdirectory in os.listdir(root):
        for _, _, files in os.walk(os.path.join(root, subdirectory)):
            filenames = [('/').join([subdirectory, f]) for f in files]
            x_col.extend(filenames)
            y_col.extend([subdirectory] * len(filenames))
    print(f'{len(x_col)} images has been read')
    df = pd.DataFrame(data=zip(x_col, y_col), columns=['filename', 'class'])
    return df


df = make_dataframe(root)

NUM_CLASSES = df['class'].nunique()
print(f'There are {NUM_CLASSES} classes in dataset')

# Creating dataloaders

# Stratified split
train_df, valid_df = train_test_split(df,
                                    test_size=0.2,
                                    shuffle=True,
                                    stratify=df['class'],
                                    random_state=0)

# class weights for sampling
map_weights = 1 / train_df['class'].value_counts()
train_df['weight'] = train_df['class'].map(map_weights)

def get_dataloaders(input_size, batch_size):
    # train dataloader with some augmentation
    train_gen = ImageDataGenerator(
                               rotation_range=20, # rotation
                               width_shift_range=0.2, # horizontal shift
                               height_shift_range=0.2, # vertical shift
                               zoom_range=0.2, # zoom
                               brightness_range = [0.6, 1.4], # brightness
                               horizontal_flip=True, # horizontal flip
                               preprocessing_function=preprocess_input)

    train_dataset = train_gen.flow_from_dataframe(train_df,
        root,
        weight_col = 'weight',
        target_size=input_size,
        shuffle=True,
        batch_size=batch_size)
    
    # validation dataloader
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    valid_dataset = valid_gen.flow_from_dataframe(valid_df,
        root,
        target_size=input_size,
        shuffle=False,
        batch_size=batch_size)
    
    return train_dataset, valid_dataset


""" Building model function """

def make_model(learning_rate=0.0001, size_inner=1024):

    base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
    pooling='avg')

    base_model.trainable = False

    #########################################

    inputs = tf.keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    base = base_model(inputs, training=False)
    x = tf.keras.layers.Dense(size_inner, activation='relu')(base)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'])
    
    return model


""" callbacks """

# Learning rate scheduler
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5,
                                              patience=5, min_lr=0.0000001, cooldown=2)
# Overfitting detection with early stopping
early_stopping  = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=15)

# saving checkpoint with best score
checkpoint = keras.callbacks.ModelCheckpoint(
    'xception.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max')


""" Setting training parameters """

size = 1024
learning_rate = 0.0001

TARGET_SIZE = (299, 299)
BATCH_SIZE = 32


""" Build model """
model = make_model(learning_rate=learning_rate, size_inner=size)
model.summary()


""" Build dataloaders """
train_dataset, valid_dataset = get_dataloaders(TARGET_SIZE, BATCH_SIZE)


""" Train model """
history = model.fit(train_dataset, epochs=50, validation_data=valid_dataset,
                    callbacks=[reduce_lr, early_stopping, checkpoint])


""" Monitoring results"""

# Calculate F1 score on validation data with best checkpoint
model = keras.models.load_model('xception.h5')

probs = model.predict(valid_dataset)
preds = np.argmax(probs, axis=1)
y_true = valid_dataset.classes

print(f1_score(y_true, preds, average='macro'))


""" Convert model to TFLite and save it"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)