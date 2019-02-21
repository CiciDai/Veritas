import io
import socket
import threading
import struct
import dlib
import cv2
import numpy as np
from PIL import Image
import os
import time
import skimage.color as Color
import h5py
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from skimage import filters, feature

newInfo = False
endSend = False
stringData = None
mutex = threading.Lock()

# load frontal face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load model
model = tf.keras.models.load_model("cnn_model216.h5")
print("Loaded model from disk")

# compile loaded model
model.compile(loss='sparse_categorical_crossentropy',
				optimizer=tf.train.GradientDescentOptimizer(0.01),
				metrics=['accuracy'])

print(model.summary())

# load the labels for emotion
emotion = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def send_process(sock):
	#send_connection = sock.makefile('wb')
	while True:
		if endSend is True:
			break
		global newInfo
		if newInfo is True:
			print("new info! Sending...")
			mutex.acquire()
			#size = len(stringData)
			#send_connection.write(struct.pack('<L', size))
			#send_connection.flush()
			#send_connection.write(stringData)
			sock.send(stringData.encode())
			mutex.release()
			print("sent!!")
			newInfo = False
	

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Start a socket listening for connections on 0.0.0.0:8000
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
recv_connection = server_socket.accept()[0].makefile('rb')
image_stream = io.BytesIO()

# Start a client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.0.0.116', 8001))
print("client connected!")
clientThread = threading.Thread(target=send_process, args=(client_socket,))
clientThread.start()

faces = []
frame_count = 0

try:
	while True:
		# Read the length of the image as a 32-bit unsigned int. If the
		# length is zero, quit the loop
		image_len = struct.unpack('<L', recv_connection.read(struct.calcsize('<L')))[0]
		if not image_len:
			break
		# Construct a stream to hold the image data and read the image
		# data from the recv_connection
		
		print("receiving data!")
		image_stream.write(recv_connection.read(image_len))
		# Rewind the stream, open it as an image with PIL and do some
		# processing on it
		image_stream.seek(0)
		img = np.asarray(Image.open(image_stream))
		
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		if frame_count >= 3:
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			#faces = detector(gray, 0)
			frame_count = 0
		top = []
		left = []
		# if there is face in frame
		if type(faces) is np.ndarray:
			print("face detected")
			face = np.zeros((faces.shape[0], 96, 96))
			print("face reshaped")
			person = 0
			for (x, y, w, h) in faces:
				print("for person " + str(person))
				left.append(x)
				top.append(y)
				print("drawing cv2 rectangle")
				#cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
				if w > h:
					s = w
				else:
					s = h
				print("cropping")
				crop = gray[y:y+s, x:x+s]
				print("resizing")
				face[person] = cv2.resize(crop, (96, 96))
				person += 1
			
			print("reshaping")
			# reshape face numpy into the shape of trained data
			face = face[:, :, :, np.newaxis]
			face = (face / 255.).astype(np.float32)
			# detect whether the image contains one of the seven emotions
			# using the trained model
			print("making prediction!")
			predicted_prob = model.predict(face)
			
			# label everyone in frame
			for p in range(person):
				print("for each person")
				prediction = predicted_prob.argmax(axis=-1)[p]
				conf_level = round(np.amax(predicted_prob[p]) * 100, 2)
				result = "Unknown"
				if conf_level > 40:
					result = str(prediction) + ":" + str(conf_level) + "%"
				print("get prediction")
				#cv2.putText(frame, emotion[prediction], (left[p], top[p]-14),
				#            cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 250), thickness=2)
				# save photo of this person's emotion
				# name = emotion[prediction] + " " + str(time.time()) + '.bmp'
				# cv2.imwrite(name, face[p] * 255)  # back to 0-255
				
				# send result over to client
				if stringData == "Unknown" and result == "Unknown":
					continue
					
				mutex.acquire()
				print("acquired mutex!")
				stringData = result
				print(stringData)
				mutex.release()
				print("released mutex!")
				newInfo = True
				
				
		#for _, d in enumerate(faces):
		#    # d is array of two corner coordinates for face bounding box
		#    # shape has 68 parts, the 68 landmarks
		#    shape = predictor(gray, d)
		#    for i in range(0, 68):
		#        x = shape.part(i).x
		#        y = shape.part(i).y
		#        cv2.circle(img, (x, y), 3, (0, 150, 255))
		#        print("ran")
		image_stream.seek(0)
		image_stream.truncate()
		frame_count += 1
		
		#cv2.imshow("img", img)#cv2.waitKey(1)
		
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break
			
finally:
	clientThread.join()
	recv_connection.close()
	server_socket.close()
	cv2.destroyAllWindows()