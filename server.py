import io
import socket
import struct
import dlib
import cv2
import numpy as np
from PIL import Image

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
image_stream = io.BytesIO()

faces = []
frame_count = 0

try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        img = np.asarray(Image.open(image_stream))
        # print(img[0])
        # print(img[0][0][0].dtype)
        # print(img.dtype)
		
        # file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # cv2.imshow("image", img)
		
        # print('Image is %dx%d' % image.size)

        # image = np.zeros(img.shape)
        # image[:, :, 0] = img[:, :, 0] / 255.0
        # image[:, :, 1] = img[:, :, 1] / 255.0
        # image[:, :, 2] = img[:, :, 2] / 255.0
        # cv2.imshow("image", image)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if frame_count >= 3:
            faces = detector(gray, 0)
            frame_count = 0
        for _, d in enumerate(faces):
            # d is array of two corner coordinates for face bounding box
            # shape has 68 parts, the 68 landmarks
            shape = predictor(gray, d)
            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(img, (x, y), 3, (0, 150, 255))
                print("ran")
        image_stream.seek(0)
        image_stream.truncate()
        frame_count += 1
        
        cv2.imshow("img", img)
        cv2.waitKey(1)
finally:
    connection.close()
    server_socket.close()
