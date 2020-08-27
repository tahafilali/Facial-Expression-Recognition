import numpy as np
import cv2 as cv
from model import FacialExpressionModel


facec = cv.CascadeClassifier('faces.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv.FONT_HERSHEY_SIMPLEX

cap = cv.VideoCapture("video.mp4")


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # Display the resulting frame
    gray_fr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow('frame', frame)
    if cv.waitKey(100) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()