import cv2, numpy
from tensorflow.keras.models import load_model

n = 0
cn = 0

model = load_model("gpt_model/school_cnn_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    # img = cv2.resize(image, (128,128))

    cv2.imshow("HI",image)
    
    if n-cn==20:
        print("ok")
        cn=n

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    n+=1

cv2.destroyAllWindows
cap.release