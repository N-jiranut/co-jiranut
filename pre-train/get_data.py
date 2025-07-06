import cv2, mediapipe, time 

ex_hand = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ex_pose = [11,12,13,14,15,16,23,24]

cap = cv2.VideoCapture(0)
pose = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()

cap = cv2.VideoCapture(0)

main = []
while True:
    ret, img = cap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image,1)
    hands_results = hands.process(image)
    pose_results = pose.process(image)
    if not ret:
        break
    
    row = []
    LH, RH, P=[], [], []
    
    if hands_results.multi_hand_landmarks:
        for id, result in enumerate(hands_results.multi_hand_landmarks):
            Hclass = hands_results.multi_handedness[id].classification[0].label
            if Hclass == "Left":
                for index, landmark in enumerate(result.landmark):
                    # print(landmark)
                    if index in ex_hand:
                        LH.append(landmark)
            else:
                for index, landmark in enumerate(result.landmark):
                    # print(landmark)
                    if index in ex_hand:
                        RH.append(landmark)
    if len(RH) == 0:
        RH=[0 for _ in range(63)]
    # if pose_results.pose_landmarkrs:
    #     landmarks = pose_results.pose_landmarks.landmark
    #     for lm in landmarks:
    #         pass
    print("===")
    cv2.imshow("Wow", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    time.sleep(1)

cv2.destroyAllWindows()
cap.release()