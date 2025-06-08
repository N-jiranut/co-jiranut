import mediapipe,cv2,time,numpy
from tensorflow.keras.models import load_model
pose_pic = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()
mpdraw = mediapipe.solutions.drawing_utils
cap = cv2.VideoCapture(0)

model = load_model("ML-model/keras_model.h5")
with open("ML-model/labels.txt", "r") as f:
    class_names = f.read().splitlines()
    
while True:
    ret, image = cap.read()
    if not ret:
        print("cant find camera")
        break
    img = cv2.flip(image, 1)       
    img=image
    results_pose = pose_pic.process(img)
    results_hand = hands.process(img)
    row=[]
    LeftHand=[]
    RightHand = []
    pose = []
    if results_hand.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
            handedness = results_hand.multi_handedness[idx].classification[0].label
            for id, lm in enumerate(hand_landmarks.landmark):
                if handedness:                        
                    if handedness == "Left" and len(LeftHand) < 42:
                        LeftHand.extend([lm.x,lm.y])
                    elif handedness == "Right" and len(RightHand) < 42:
                        RightHand.extend([lm.x,lm.y])
    if len(LeftHand) == 0:
        LeftHand=[0 for n in range(42)]     
    if len(RightHand) == 0:
        RightHand=[0 for n in range(42)]     
    row.extend(LeftHand)
    row.extend(RightHand)
    
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        for lm in landmarks:
            pose.extend([lm.x, lm.y, lm.z])
    if len(pose) == 0:
        pose=[0 for n in range(99)]  
    row.extend(pose)    
    
    if len(row) == 183:
        landmarks_np = numpy.array(row).reshape(1, -1)
        pred = model.predict(landmarks_np)
        index = numpy.argmax(pred)
        label = class_names[index]
        print("Prediction:", label, "Confidence:", pred[0][index])
    else:
        print(len(row))
    
    cv2.imshow("Cheese",img)
    cv2.waitKey(1)
    time.sleep(.1)
    
cap.release()
cv2.destroyAllWindows()