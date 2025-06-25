import cv2, mediapipe, glob, pandas

name="test"

file = glob.glob('C:/Users/Thinkpad/Documents/co-jiranut/for-Jak/current_picture/*.jpg')

pose_pic = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()
mpdraw = mediapipe.solutions.drawing_utils
# cap = cv2.VideoCapture(0)

data=[]

for pic in file:
    img = cv2.imread(pic)
    img = cv2.flip(img, 1)       
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
    data.append(row)
df = pandas.DataFrame(data)

df.to_csv(f"C:/Users/Thinkpad/Documents/co-jiranut/for-Jak/csv_file/test-{name}.csv",index = False)