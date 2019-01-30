import dlib
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

frame_rate_calc = 1
freq = cv2.getTickFrequency()

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
if __name__ == "__main__":

    if camera_type == 'picamera':
        # Initialize Picamera and grab reference to the raw capture
        camera = PiCamera()
        camera.resolution = (IM_WIDTH, IM_HEIGHT)
        camera.framerate = 30
        rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
        rawCapture.truncate(0)
        for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            t1 = cv2.getTickCount()
            frame = frame1.array
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

            cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Facial Landmarks', frame)

            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

            rawCapture.truncate(0)

        camera.close()

    elif camera_type == 'usb':
        cap = cv2.VideoCapture(0)
        ret = cap.set(3, IM_WIDTH)
        ret = cap.set(4, IM_HEIGHT)
        while(True):

            t1 = cv2.getTickCount()

            ret,frame = cap.read()
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
                    # cv2.putText(frame, str(i), (x+2, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 200), 1)

            cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Facial Landmarks', frame)

            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()



