import cv2
def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(0)
    img_id = 0
# 
    while True:
        ret,frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1   
            face = cv2.resize(face_cropped(frame),(250,250))
            file_name_path = "mirza/" + "mirza_" + str(img_id) +".jpg"
            cv2.imwrite(file_name_path,face)
            cv2.imshow("cropped face",face)
            if cv2.waitKey(1) == 13 or int (img_id) == 100 :
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("completed")

generate_dataset()