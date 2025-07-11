import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)  # Height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)   # Width

a = 2000

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
while True:
    ret, image = cap.read()
    if not ret:
        break
    
    
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # results = hands.process(image)
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         image_height, image_width, _ = image.shape
    #         x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    #         y_coords = [landmark.y for landmark in hand_landmarks.landmark]
    #         x_min = int(min(x_coords) * image_width)
    #         x_max = int(max(x_coords) * image_width)
    #         y_min = int(min(y_coords) * image_height)
    #         y_max = int(max(y_coords) * image_height)
    #         image = image[y_min:y_max, x_min:x_max]

    # print(image.shape)
    image1 = image[100:a, 100:200]
    cv2.imshow('Cropped Hand', image)
    cv2.imshow('image', image1)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
cap.release   
