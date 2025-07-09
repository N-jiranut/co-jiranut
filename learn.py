import cv2
import numpy as np
import mediapipe as mp

hands = mp.solutions.hands.Hands()
mpdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

height = 640
width = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Height

# black_screen = np.zeros((width, height, 3), dtype=np.uint8)

# x, y = 10, 10
# cv2.circle(black_screen, (x, y), radius=1, color=(0, 0, 255), thickness=7)

# cv2.imshow("test", black_screen)

# cv2.waitKey(0)


while True:
    black_screen = np.zeros((width, height, 3), dtype=np.uint8)
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    hand_result = hands.process(img)

    if hand_result.multi_hand_landmarks:
        # print("Yes")
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                # handedness = hand_result.multi_handedness[idx].classification[0].label
                for id, lm in enumerate(hand_landmarks.landmark): 
                    mp.solutions.drawing_utils.draw_landmarks(black_screen,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)

    if not ret:
        print("Cam not found.")
        break
    cv2.imshow("Frame", black_screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows
# cap.release
print('end')