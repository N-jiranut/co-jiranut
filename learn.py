import cv2
import numpy as np
import mediapipe as mp

hands = mp.solutions.hands.Hands()
mpdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
focus = 9

# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)  # Height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)   # Width

# while True:
    # ret, img = cap.read()
    # cv2.imshow("test", img)
    # if cv2.waitKey(1) == ord("q"):
    #     break

while True:
    landmark_location = []
    black_screen = np.zeros((640, 480, 3), dtype=np.uint8)
    ret, img = cap.read()
    if not ret:
        print("Cam not found.")
        break
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hand_result = hands.process(img)
    
    if hand_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                handedness = hand_result.multi_handedness[idx].classification[0].label
                if handedness == "Right":
                    for id, lm in enumerate(hand_landmarks.landmark): 
                        x, y, z = lm.x, lm.y, lm.z
                        landmark_location.append([x,y])                        
                        mp.solutions.drawing_utils.draw_landmarks(black_screen,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)

    if len(landmark_location)>0:     
            
        centerx = round(landmark_location[focus][0]*400)
        centery = round(landmark_location[focus][1]*400)
        mx = centerx+200
        nx = centerx-200
        my = centery+200
        ny = centery-200
 
        if mx > 680:
            mx = 680
        if nx < 0:
            nx = 0
        if my > 480:
            my = 480
        if ny < 0:
            ny = 0
            
        # cimg = img[100:200, 100:200]
        cimg = black_screen[ny:my, nx:mx]
        cv2.imshow("Crop", cimg)  
                 
            # loss_x, loss_y = landmark_location[focus]
            # landmark_location[focus] = [200,200]
            # colorb = 0
            # colorg = 0
            # colorr = 255
            # for id_loss, items in  enumerate(landmark_location):
            #     if id_loss == focus:
            #         continue
            #     new_x, new_y = landmark_location[id_loss]
            #     new_x = round((new_x/loss_x) * 200)
            #     new_y = round((new_y/loss_y) * 200)
            #     landmark_location[id_loss] = [new_x, new_y]
                # cv2.circle(black_screen, [new_x,new_y], 1, (255,0,0), 7)
        # mp.solutions.drawing_utils.draw_landmarks(black_screen,landmark_location,mp.solutions.hands.HAND_CONNECTIONS)
            # for location in landmark_location:
            #     cv2.circle(black_screen, location, 1, (colorb,colorg,colorr), 7)
            #     colorr+=10
            #     colorg+=50
            #     colorb+=20
            #     if colorr>255:
            #         colorr=0
            #     if colorb>255:
            #         colorb=0
            #     if colorg>255:
            #         colorg=0
           
        # cv2.imwrite(f"teachable_machine/test/saved2.jpg",black_screen) 
     
    cv2.imshow("Real", img)        
    # cv2.imshow("Frame", black_screen)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows
cap.release
print('end')