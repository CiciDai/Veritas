import dlib
import cv2

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame')
    while(True):
        _,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 1)
        for _, d in enumerate(faces):
            # d is array of two corner coordinates for face bounding box
            # shape has 68 parts, the 68 landmarks
            shape = predictor(frame, d)
            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(frame, (x, y), 3, (0, 150, 255))
                cv2.putText(frame, str(i), (x+2, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 200), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

