# import random,numpy,sklearn
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

print("Starting...")
import tensorflow
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import catagorical_crossentropy

model = tensorflow.keras.models.Sequential([
    Dense(units=16, input_shape=(1,), activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=2, activation="softmax")
])
print("It work!")