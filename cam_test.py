import cv2, mediapipe

cap = cv2.VideoCapture(0)
pose = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()

while True:
    ret, image = cap.read()
    if not ret:
        print("Cant find camera")
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image,1)
    results_hands = hands.process(image)
    results_pose = pose.process(image)

    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[idx].classification[0].label
            for id, lm in enumerate(hand_landmarks.landmark):
                if handedness:   
                    mediapipe.solutions.drawing_utils.draw_landmarks(image,hand_landmarks,mediapipe.solutions.hands.HAND_CONNECTIONS)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("test na ja", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()