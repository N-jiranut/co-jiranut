import cv2, mediapipe, time
pose = mediapipe.solutions.pose.Pose()
hands = mediapipe.solutions.hands.Hands()
mpdraw = mediapipe.solutions.drawing_utils
cap = cv2.VideoCapture(0)
current_time = 0
n=0
now=0
work = False

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
        print(n)
        # path = f'current_picture/captured_image{n}.jpg'
        # cv2.imwrite(f'current_picture/captured_image{n}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(f'current_picture/image{n}.jpg', img)
    
    key = cv2.waitKey(1)

    if key == ord("s") and work == False:
        print('starting')
        n=0
        work = True
        now = current_time

    if current_time-now==100:
        work = False
        print("end")

    if key == ord("q"):
        break
        
    n+=1

    cv2.imshow("Sample",img)
    # time.sleep(.1)

cap.release()
cv2.destroyAllWindows()