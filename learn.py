import cv2
cap = cv2.VideoCapture(0)

for i in range(10):
    ret, img = cap.read()
    if ret:
        cv2.imwrite(f"for-Jak/current_picture/image{i}.jpg", img)