# from tensorflow.keras import layers, models

# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=10,
#           validation_data=(test_images, test_labels))

# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical

# 1. Load
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# print(train_labels)
# # 2. Normalize
# train_images = train_images.astype('float32') / 255.0
# test_images  = test_images.astype('float32') / 255.0

# # 3. Encode labels
# train_labels = to_categorical(train_labels, num_classes=10)
# test_labels  = to_categorical(test_labels, num_classes=10)

# print(len(train_labels))

# print(train_images.shape, train_labels.shape)
# # → (50000, 32, 32, 3) (50000, 10)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# Load dataset
dataset_dir = 'data/'  # change to your folder path
img_size = (224, 224)
batch_size = 8

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical'  # one-hot encoded labels
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode='categorical'
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Pretrained base model
base_model = MobileNetV2(input_shape=img_size + (3,),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # freeze base

# Model building
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
