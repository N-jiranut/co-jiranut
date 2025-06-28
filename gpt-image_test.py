import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. Load
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 2. Normalize
# train_images = train_images.astype('float32') / 255.0
# test_images  = test_images.astype('float32') / 255.0

# 3. Encode labels
# train_labels = to_categorical(train_labels, num_classes=10)
# test_labels  = to_categorical(test_labels, num_classes=10)

# print(train_images.shape, train_labels.shape)
# # → (50000, 32, 32, 3) (50000, 10)

# img_height, img_width = 150, 150
# batch_size = 32

# Load and augment training data
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_gen = train_datagen.flow_from_directory(
#     'gpt-image/train',
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical'  # or 'binary' for 2 classes
# )

# Validation data
# val_datagen = ImageDataGenerator(rescale=1./255)
# val_gen = val_datagen.flow_from_directory(
#     'gpt-image/validation',
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical'
# )

datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values (0–255 → 0–1)
    rotation_range=20,      # Randomly rotate images
    zoom_range=0.2,         # Random zoom
    horizontal_flip=True,   # Randomly flip images
    validation_split=0.2    # Split dataset for training/validation
)

train_data = datagen.flow_from_directory(
    'gpt-image',
    target_size = (128,128),
    batch_size=16,          
    class_mode='categorical',
    subset='training')

val_data = datagen.flow_from_directory(
    'gpt-image',
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# train_dataset = tf.keras.utils.image_dataset_from_directory(
#     'gpt-image',
#     labels='inferred',
#     label_mode='int',  # Options: 'int', 'categorical', 'binary'
#     batch_size=32,
#     image_size=(150, 150),  # Adjust based on your model's requirements
#     shuffle=True,
#     seed=123
# )

# print("VVV")
# print(train_dataset.value)

# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])


early_stop = EarlyStopping(
    monitor='val_loss',      # What to monitor (usually 'val_loss' or 'val_accuracy')
    patience=3,              # How many epochs to wait before stopping after no improvement
    restore_best_weights=True  # Restore model weights from the best epoch
)


# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(3, activation='softmax')  # dynamic class count
# ])

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')  # one output per class
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_dataset, epochs=10)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stop]
)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

choice = input("Wanna save? >:")
if choice == "0":
    model.save("gpt_model/school_cnn_model.h5")