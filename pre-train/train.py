import cv2
import sys
import os
from openpose import pyopenpose as op

# Params
params = {
    "model_folder": "models/",
    "hand": True,
    "body": 1,
    "face": False,  # Set True if you want face too
    "hand_detector": 2,  # Use body keypoints to find hands
}

# Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Load image
imagePath = "your_image.jpg"
imageToProcess = cv2.imread(imagePath)

# Process
datum = op.Datum()
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum])

# Display result
cv2.imshow("OpenPose Result", datum.cvOutputData)
cv2.waitKey(0)
