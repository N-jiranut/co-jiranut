import cv2, mediapipe, time
pose = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()
mpdraw = mediapipe.solutions.drawing_utils
cap = cv2.VideoCapture(0)
current_time = 0
n=0
now=0
work = False
prepare = False

frame = 100

while True:
    current_time+=1
    ret, image = cap.read()
    if not ret:
        print("cant find camera")
        break
    img = cv2.flip(image, 1)

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mpdraw.draw_landmarks(img,results.pose_landmarks,mediapipe.solutions.pose.POSE_CONNECTIONS)

    results_hand = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results_hand.multi_hand_landmarks:
        for handLms in results_hand.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handLms, mediapipe.solutions.hands.HAND_CONNECTIONS)
            
    if work == True:
        print(f"pic-{n-20}")
        cv2.imwrite(f'C:/Users/Thinkpad/Documents/co-jiranut/for-Jak/current_picture/image{n}.jpg', img)
    
    key = cv2.waitKey(1)

    if key == ord("s") and work == False:
        print('starting')
        n=0
        prepare = True
        now = current_time

    if prepare == True and n==0:
        print("Ready!")
    if prepare == True and n ==20:
        print("Cupturing...")
        work=True
        prepare=False

    if current_time-now==frame+20:
        work = False
        print("end")

    if key == ord("q") and work == False:
        break
        
    n+=1

    cv2.imshow("Sample",img)

cap.release()
cv2.destroyAllWindows()