import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model('my_model.h5')
from sklearn.preprocessing import LabelEncoder

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy')

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Resize face ROI to match input size of model
        face_roi = cv2.resize(face_roi, (100, 100))

        # Convert face ROI to array
        face_roi = np.array(face_roi, dtype=np.float32)

        # Normalize array
        face_roi /= 255.0

        # Reshape array to match input shape of model
        face_roi = face_roi.reshape(1, 100, 100, 1)

        # Predict label of face
        label = label_encoder.inverse_transform(model.predict(face_roi))[0]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write predicted label above rectangle
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Menampilkan frame
    cv2.imshow('Face Recognition', frame)

    # Tombol q untuk keluar
    if cv2.waitKey(1) == ord('q'):
        break

# Stop kamera dan tutup semua window
cap.release()
cv2.destroyAllWindows()