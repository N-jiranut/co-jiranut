# import random,numpy,sklearn
# import sklearn.metrics
# import sklearn.preprocessing
# samples = []
# labels = []
# for i in range(5):
#     young = random.randint(13, 64)
#     samples.append(young)
#     labels.append(1)
# for i in range(5):
#     old = random.randint(65, 100)
#     samples.append(old)
#     labels.append(0)
# for i in range(100):
#     young = random.randint(13, 64)
#     samples.append(young)
#     labels.append(0)
# for i in range(100):
#     old = random.randint(65, 100)
#     samples.append(old)
#     labels.append(1)

# train_samples = numpy.array(samples)
# train_labels = numpy.array(labels)
# suff = numpy.random.permutation(len(train_labels))
# train_labels, train_samples = train_labels[suff], train_samples[suff]

# scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
# scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# print("Starting...")
import tensorflow
from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import categorical_crossentropy

# model = tensorflow.keras.models.Sequential([
#     Dense(units=16, input_shape=(1,), activation="relu"),
#     Dense(units=32, activation="relu"),
#     Dense(units=2, activation="softmax")
# ])
# # print("It work!")
# # model.summary()

# model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # print("OKk")

# model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=40, shuffle=True, verbose=2)

# print("Complete model")

# import matplotlib.pyplot as plt
# import itertools

# prediction = model.predict(x=scaled_train_samples, batch_size=10, verbose=0)
# rounded_prediction = numpy.argmax(prediction, axis=-1)
# print(rounded_prediction)

# cm = sklearn.metrics.confusion_matrix(y_true=train_labels, y_pred=rounded_prediction)

# def plot_confusion_matric(cm, classes, normalize=False, title="Confusion matric", cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title=title
#     plt.colorbar()
#     trick_marks=numpy.arange(len(classes))
#     plt.xticks(trick_marks, classes, rotation=45)
#     plt.yticks(trick_marks, classes)
    
#     if normalize:
#         cm=cm.astype('float')/cm.sum(axis=1)[:,numpy.newaxis]
#         print("Normalize matric")
#     else:
#         print("Not normalize matric")
    
#     print(cm)
    
#     thresh=cm.max()/2.
#     for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i,j], horizontalalignment="center", color="white" if cm[i,j]>thresh else "black")
    
#     plt.tight_layout()
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")
    
# cm_plot_label=["no_side_effects","had_side_effects"]
# plot_confusion_matric(cm=cm, classes=cm_plot_label, title="confusion_matric")
# print("Graph ploting")

# model.save("ML-model/medical_train_model.h5")

# my_model = tensorflow.keras.models.load_model("ML-model/medical_train_model.h5")

model2 = tensorflow.keras.models.Sequential([
    Dense(units=16, input_shape=(1,), activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=2, activation="softmax")
])
model2.load_weights("ML-model/medical_train_model.h5")