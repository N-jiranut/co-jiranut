import cv2, numpy
from tensorflow.keras.models import load_model

n = 0
cn = 0

model = load_model("gpt_model/school_cnn_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    # img = cv2.resize(image, (128,128))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img / 255
    # img = numpy.expand_dims(img, axis=0)
    
    
    if n-cn==20:
        img = cv2.resize(image, (128,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = numpy.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_index = numpy.argmax(prediction[0])
        class_labels = ["five", 'two', 'zero']
        print("Predicted class:", class_labels[predicted_index])
        cn=n

    cv2.imshow("HI",image)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    n+=1

cv2.destroyAllWindows
cap.release