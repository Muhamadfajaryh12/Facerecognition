import cv2
import os
cap = cv2.VideoCapture(0)
dataset_folder = "datasetFajar/"
my_name = "Fajar"
# os.mkdir(dataset_folder + my_name)
num_sample = 100

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret :
        i += 1    
        cv2.imshow("Capture Photo", frame)
        cv2.imwrite("dataset/%s/%s_%04d.jpg" %  (my_name, my_name, i), cv2.resize(frame, (250,250)))
        
        if cv2.waitKey(1) == ord('q') or i == num_sample:
            break
cap.release()
cv2.destroyAllWindows()