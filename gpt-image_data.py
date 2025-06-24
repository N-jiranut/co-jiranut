import cv2,pandas,numpy

cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    
    cv2.putText(img, "Hello!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("test",img)
    resized = cv2.resize(img, (480, 240))
    cv2.imshow("tests",resized)
    
    key = cv2.waitKey(1)
    cv2.imshow("test",img)
    if key == ord("q"):
        break
print("end")
cv2.destroyAllWindows
cap.release