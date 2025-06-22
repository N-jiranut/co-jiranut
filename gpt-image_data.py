import cv2,pandas,numpy

cap = cv2.VideoCapture(0)

pic = 20

img = cap.read()

# n = numpy.array(img)

print(img[1])

# for i in range(pic):
#     img = cap.read()
#     print(img.shape)
print("end")