from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. Load
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 2. Normalize
train_images = train_images.astype('float32') / 255.0
test_images  = test_images.astype('float32') / 255.0

# 3. Encode labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels  = to_categorical(test_labels, num_classes=10)

# print(len(train_labels))

# print(train_images.shape, train_labels.shape)
# # → (50000, 32, 32, 3) (50000, 10)

img_height, img_width = 150, 150
batch_size = 32

# Load and augment training data
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
    'gpt-image/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # or 'binary' for 2 classes
)

# Validation data
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    'gpt-image/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')  # dynamic class count
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=10, validation_data=val_gen)