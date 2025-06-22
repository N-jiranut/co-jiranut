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

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. Load
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_labels)
# # 2. Normalize
# train_images = train_images.astype('float32') / 255.0
# test_images  = test_images.astype('float32') / 255.0

# # 3. Encode labels
# train_labels = to_categorical(train_labels, num_classes=10)
# test_labels  = to_categorical(test_labels, num_classes=10)

# print(len(train_labels))

# print(train_images.shape, train_labels.shape)
# # → (50000, 32, 32, 3) (50000, 10)
