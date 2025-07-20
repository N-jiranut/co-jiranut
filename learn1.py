# from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import mediapipe as mp
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# # Load the model
import tensorflow as tf
from tensorflow.keras.models import load_model

# ลองสร้างคลาส DepthwiseConv2D ที่ "ยอมรับ" groups
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs: # ลบ groups ออกไปก่อนส่งให้ constructor จริง
            del kwargs['groups']
        super().__init__(*args, **kwargs)

try:
    model = load_model("teachable_machine/converted_keras/keras_Model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
# model = load_model("teachable_machine/converted_keras/keras_Model.h5", compile=False)
# # Load the labels
class_names = open("teachable_machine/converted_keras/labels.txt", "r").readlines()

# # CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()

while True:
    # Grab the webcamera's image.

    black_screen = np.zeros((224, 224, 3), dtype=np.uint8)
    ret, img = camera.read()
    if not ret:
        print("Cam not found.")
        break
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hand_result = hands.process(img)
    
    if hand_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            handedness = hand_result.multi_handedness[idx].classification[0].label
            if handedness == "Right":
                for id, lm in enumerate(hand_landmarks.landmark):                     
                    mp.solutions.drawing_utils.draw_landmarks(black_screen,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
    cv2.imshow("canvas", black_screen)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(black_screen, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

# camera.release()
# cv2.destroyAllWindows()