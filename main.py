import cv2
import pickle
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
label_dict = {'Name', 1}
with open("labels.pickle", 'rb') as f:
    original_dict = pickle.load(f)
    label_dict = {v:k for k,v in original_dict.items()}
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
while(ret):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        # img = '4.jpg'
        # cv2.imwrite(img, roi_color)

        id, conf = recognizer.predict(roi_gray)
        if conf>=50 and conf<=95:
            print(id)
            print(label_dict[id])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = label_dict[id]
            cv2.putText(frame, name, (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        eyes = eye_cascade.detectMultiScale(gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


    cv2.imshow('Frame', frame)
    if (cv2.waitKey(20)==27):
        break
cap.release()
cv2.destroyAllWindows()