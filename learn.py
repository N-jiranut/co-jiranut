import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe

hands = mediapipe.solutions.hands.Hands()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
while True:
    success, img = cap.read()
    # hands, img = detector.findHands(img)
    img = cv2.flip(img, 1)
    result_hands = hands.process(img)

    if result_hands:
        print(result_hands.multi_hand_landmarks)
        # print(hands[0]["lmList"][0],"\n\n")
        # cv2.circle(img, hands[0]["lmList"][0][0:2], 0, (0,0,0), 10)
    # print(img)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    time.sleep(.5)
cap.release
cv2.destroyAllWindows