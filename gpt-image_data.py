import cv2,time

cap = cv2.VideoCapture(0)

classs = ["zero","two","five"]

for loop in range(3):
    print(f"Round {loop+1}")
    print(classs[loop])
    while True:
        ret, img = cap.read()   
        cv2.imshow("test",img)
        
        key = cv2.waitKey(1)
        if key == ord("s"):
            print("yes")
            for i in range(20):
                cv2.imwrite(f"gpt-image/{classs[loop]}/image{i}.jpg", img)
            print("Success")
            break

print("end")
cv2.destroyAllWindows
cap.release