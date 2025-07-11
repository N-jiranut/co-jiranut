import cv2, mediapipe, time 

take_hand = [0,10]

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
    results_pose = pose.process(image)
    results_hands = hands.process(image)

    if results_hands.multi_hand_landmarks:
        test=[]
        # print(results_hands.multi_hand_landmarks.landmark)
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # handedness = results_hands.multi_handedness[idx].classification[0].label
            # test.append(hand_landmarks.landmark)
            # print(len(test))
            
            print(type(hand_landmarks.landmark))
            
            # break
            # if handedness:   
            #     mediapipe.solutions.drawing_utils.draw_landmarks(image,hand_landmarks,mediapipe.solutions.hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                if id in take_hand:
                    print(id)
                
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("test na ja", image)
    
    print("==========")
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    time.sleep(0)
cap.release()
cv2.destroyAllWindows()