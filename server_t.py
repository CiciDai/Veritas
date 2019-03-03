import io
import socket
import threading
from multiprocessing import Process, Queue, Lock, Value
#import queue
#from collections import deque
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
	counter = 0
	while True:
		print("send_process running")
		if endSend is True:
			print("end sending!")
			sock.send(str("end_stream").encode())
			break
		#global newInfo
		#if newInfo is True:
		if resultQueue.empty() is False:
			print("resultQueue not empty")
			#size = len(stringData)
			#send_connection.write(struct.pack('<L', size))
			#send_connection.flush()
			#send_connection.write(stringData)
			resultMutex.acquire()
			sock.send(resultQueue.get().encode())
			resultMutex.release()
			print("sent frame " + str(counter) + " at time " + str(time.time()))
			counter += 1
			#newInfo = False
	
def assignJob(imgQueue, resultQueue, imgMutex, resultMutex, end):
	#while endSend is False or imgQueue.empty() is False:
	while end.value == 0 or imgQueue.empty() is False:
	#if imgQueue.empty() is False:
		imgMutex.acquire()
		if imgQueue.empty() is True:
			imgMutex.release()
		else:
			img = imgQueue.get()
			print("got an image to process")
			imgMutex.release()
			results = processImg(img)
			resultMutex.acquire()
			for result in results:
				resultQueue.put(result)
			resultMutex.release()
	print(str(end.value) + " process assignJob ending")

def processImg(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	#if frame_count >= 3:
	#	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30,30))
	#	#faces = detector(gray, 0)
	#	frame_count = 0
	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30,30))
	
	top = []
	left = []
	results = []
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
			result = "8::"
			if conf_level > 40:
				result = str(prediction) + ":" + str(conf_level) + "%:"
			print("get prediction")
			#cv2.putText(frame, emotion[prediction], (left[p], top[p]-14),
			#            cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 250), thickness=2)
			# save photo of this person's emotion
			# name = emotion[prediction] + " " + str(time.time()) + '.bmp'
			# cv2.imwrite(name, face[p] * 255)  # back to 0-255
			
			# send result over to client
			
			#if stringData == "Unknown" and result == "Unknown":
			#	continue
				
			#resultMutex.acquire()
			#print("acquired mutex!")
			#resultQueue.put(result)
			print(result)
			results.append(result)
			#resultMutex.release()
			#print("released mutex!")
			#newInfo = True
	return results


if __name__ == '__main__':

	newInfo = False
	endSend = False
	stringData = None
	imgQueue = Queue()
	imgMutex = Lock()
	resultQueue = Queue()
	end = Value('i', 0)
	resultMutex = Lock()


	#predictor_path = 'shape_predictor_68_face_landmarks.dat'
	#detector = dlib.get_frontal_face_detector()
	#predictor = dlib.shape_predictor(predictor_path)

	workerProcesses = []
	for i in range(10):
		workerProcesses.append(Process(target=assignJob, args=(imgQueue, resultQueue, imgMutex, resultMutex, end)))
		workerProcesses[i].start()
		print("started workers")
		
	# Start a socket listening for connections on 0.0.0.0:8000
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind(('0.0.0.0', 8000))
	server_socket.listen(0)
	
	# Accept a single connection and make a file-like object out of it
	recv_connection = server_socket.accept()[0].makefile('rb')
	image_stream = io.BytesIO()
	
	# Start a client socket
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client_socket.connect(('10.18.233.18', 8001))
	print("client connected!")
	clientThread = threading.Thread(target=send_process, args=(client_socket,))
	clientThread.start()
	
	faces = []
	frame_count = 0
	count = 0
	
	
	try:
		while True:
			# Read the length of the image as a 32-bit unsigned int. If the
			# length is zero, quit the loop
			image_len = struct.unpack('<L', recv_connection.read(struct.calcsize('<L')))[0]
			if image_len == 0:
				# tell client to end
				print("rec'd 0 size end string")
				#endSend = True
				end.value = 1
				break
			# Construct a stream to hold the image data and read the image
			# data from the recv_connection
			
			print("receiving data!")
			image_stream.write(recv_connection.read(image_len))
			print("received frame #" + str(count) + " at time = " + str(time.time()))
			count += 1
			# Rewind the stream, open it as an image with PIL and do some
			# processing on it
			image_stream.seek(0)
			img = np.asarray(Image.open(image_stream))
			
			imgQueue.put(img)
	
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
		for worker in workerProcesses:
			worker.join()
		clientThread.join()
		recv_connection.close()
		server_socket.close()
		cv2.destroyAllWindows()