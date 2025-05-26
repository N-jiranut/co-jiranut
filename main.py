import pandas, mediapipe, cv2, numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# ======================config==================================
picture_file_name="test_house"
number_of_picture=50
label="no person"
model_name="test_sigma" #config after collect picture
# ==============================================================

path=f"data/{picture_file_name}.csv"

def take_photo(path,frame,label):
    pose_pic = mediapipe.solutions.pose.Pose()
    hands = mediapipe.solutions.hands.Hands()
    cap = cv2.VideoCapture(0)
    data=[]
    for _ in range(frame):
        ret, image = cap.read()
        if not ret:
            print("cant find camera")
            break
        img = cv2.flip(image, 1)       
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
        row.append(label)    
        data.append(row)
        
    df = pandas.DataFrame(data)    
    df.to_csv(path, mode='a', index=False, header=False)
    
def train_model(name):
    try:
        df = pandas.read_csv(path)
    except:
        print("\n\n\n------>Cant find data file<------\n\n\n")
    else:
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        if input("Type 1 to save model and 0 to skip:") == "1":
            joblib.dump(model, f'ML-model/{name}.pkl')
            print("\n\n\nsave complete\n\n\n")
        
def use_model():
    pass
    try:
        model = joblib.load(f'ML-model/{model_name}.pkl')
    except:
        print("\n\n\n------>cant find model<------\n\n\n")
    else:
        pose = mediapipe.solutions.pose.Pose()
        hands = mediapipe.solutions.hands.Hands()
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            results = pose.process(image)
            results_hand = hands.process(image)
            
            LeftHand=[]
            RightHand = []
            pose_data = []
            row=[]
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
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for lm in landmarks:
                    pose_data.extend([lm.x, lm.y, lm.z])
            if len(pose_data) == 0:
                pose_data=[0 for n in range(99)]  
            row.extend(pose_data)        
            X_input = numpy.array(row).reshape(1, -1)
            prediction = model.predict(X_input)
            cv2.putText(frame, f'Pose: {prediction[0]}', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Pose Classification', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

while True:
    print("Type '0' to collect picture, '1' to start train model, and '2' to use model or 'exit' to exit")
    choice = input("Type here>:")
    if choice == "0":
        take_photo(path, number_of_picture, label)
    elif choice == "1":
        train_model(model_name)
    elif choice == "2":
        use_model()
    elif choice == "exit":
        break
    else:
        print("Are you serious?")
print("Process end")