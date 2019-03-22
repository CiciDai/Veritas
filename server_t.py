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

import NLP


ospath = "C:\Workspace\Veritas\Veritas"
# load frontal face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load model
#model = tf.keras.models.load_model("all_model.tflite")
print("Loaded model from disk")

interpreter = tf.lite.Interpreter(model_path=os.path.join(ospath, "all_model.tflite"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# compile loaded model
#model.compile(loss='sparse_categorical_crossentropy',
#			optimizer=tf.train.GradientDescentOptimizer(0.01),
#			metrics=['accuracy'])

#print(model.summary())
	
# load the labels for emotion
emotion = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
sentiment = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']

currentFrame = None
circBuffer = np.zeros((150,3), dtype=int)
NLPQueue = Queue()


contrastMatrix = [[0.6, 0.7, 0.7, 0.5, 0.2, 0.5, 0.5],
					[0.0, 0.3, 0.8, 0.7, 1.0, 0.5, 0.7],
					[0.2, 0.3, 0.7, 0.8, 0.9, 0.9, 0.8],
					[0.3, 0.0, 0.6, 0.3, 1.0, 0.8, 0.3],
					[0.8, 0.6, 0.0, 0.2, 1.0, 0.3, 0.2],
					[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
					[0.5, 0.5, 0.3, 0.0, 1.0, 0.0, 0.0],
					[0.5, 0.3, 0.3, 0.5, 0.5, 0.5, 0.3]]



#def predictLies(emotion):
#	total = 0
#	for i in range(len(circBuffer)):
#		total += contrastMatrix[emotion][circBuffer[i]]
#	total /= len(circBuffer)
#	if total > 0.6:
#		print("lie detected")

def getStrTime():
	msTime = str(round(time.time() * 1000))
	return msTime[-8:]


def send_process(sock, endSend, resultQueue):
	#send_connection = sock.makefile('wb')
	counter = 0
	while True:
		if endSend is True:
			print("end sending!")
			sock.send(str("end_stream").encode())
			break
		#global newInfo
		#if newInfo is True:
		if resultQueue.empty() is False:
			#size = len(stringData)
			#send_connection.write(struct.pack('<L', size))
			#send_connection.flush()
			#send_connection.write(stringData)
			#resultMutex.acquire()
			result = resultQueue.get()
			global circBuffer
			circBuffer = np.roll(circBuffer, 1, axis=0)
			circBuffer[0][0] = result[0]
			circBuffer[0][1] = int(result[1].split(':')[0])
			print("split result = " + (result[1].split(':')[1])[:-1])
			circBuffer[0][2] = float((result[1].split(':')[1])[:-1])
			#print(circBuffer)
			sock.send(result[1].encode())
			#resultMutex.release()
			print("sent frame " + str(counter) + ": " + result[1] + " at time = " + str(time.time()))
			counter += 1
			#newInfo = False
			
	
def assignJob(imgQueue, resultQueue, imgMutex, faceQueue, end, framesProcessed, frame_num, threadNum):
	#while endSend is False or imgQueue.empty() is False:
	while end.value == 0 or imgQueue.empty() is False:
	#if imgQueue.empty() is False:
		# if framesTaken.value % frame_num == threadNum: # my turn
			# imgMutex.acquire()
			# if imgQueue.empty() is True:
				# imgMutex.release()
			# else:
				# timeImg = imgQueue.get()
				# imgMutex.release()
				# framesTaken.value += 1
				
				# face = processImg(timeImg[1])
				# print("processed frame " + str(framesProcessed.value) + " at time = " + str(time.time()))
					
				# if face is not None:
					# while framesProcessed.value % frame_num != threadNum: # wait for your turn
						# continue
					# faceQueue.put((timeImg[0], face))
				# framesProcessed.value += 1
				# if faceQueue.qsize() >= frame_num:
					# continue
		imgMutex.acquire()
		if imgQueue.empty() is True:
			imgMutex.release()
		else:
			timeImg = imgQueue.get()
			imgMutex.release()
				
			face = processImg(timeImg[1])
			print("processed frame " + str(framesProcessed.value) + " at time = " + str(time.time()))
					
			if face is not None:
				faceQueue.put((timeImg[0], face))
			framesProcessed.value += 1
			if faceQueue.qsize() >= frame_num:
				continue
				
	print(str(end.value) + " process assignJob ending")

def processImg(img):
	#print("processing image")
	processStart = time.time()
	faces = face_cascade.detectMultiScale(img, 1.3, 5, minSize=(50,50))
	
	# if there is face in frame
	if type(faces) is np.ndarray:
		face = np.zeros((96, 96))
		max_area = 0
		for (x, y, w, h) in faces:
			if max_area >= (w * h):
				continue
			
			max_area = w * h
			if w > h:
				s = w
			else:
				s = h
			crop = img[y:y+s, x:x+s]
			gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
			#gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
			#face[person] = cv2.resize(gray, (96, 96))
			face = cv2.resize(gray, (96, 96))

		# reshape face numpy into the shape of trained data
		face = face[:, :, np.newaxis]
		
		return face
	#print("processing took " + str(time.time() - processStart) + "seconds.")
	return None

	
def predictImg(faceQueue, faceMutex, resultQueue, frame_num, end):
	while end.value == 0:
		faceMutex.acquire()
		if faceQueue.qsize() >= frame_num:
			faces = np.zeros((frame_num, 96, 96, 1))
			timestamp = 0
			timeFace = None
			for i in range(frame_num):
				timeFace = faceQueue.get()
				faces[i] = timeFace[1]
			faceMutex.release()
			timestamp = timeFace[0]
			
			if len(faces) == 0:
				print("got empty faces list")
				continue
			
			#faces = np.array(faces)
			predicted_prob = np.zeros((2, 8))
			
			#input_data = faces[0][tf.newaxis, ...]
			for i in range(2):
				input_data = np.array(faces[i*5][tf.newaxis,...], dtype=np.float32)
				#print(input_data.shape)
				#print(input_data)
				interpreter.set_tensor(input_details[0]['index'], input_data)
				interpreter.invoke()
				prob = interpreter.get_tensor(output_details[0]['index'])
				#print(prob)
				predicted_prob[i] = prob
			
			
			##predicted_prob = model.predict(faces)  # predict
			#print("predicted_prob:")
			#for p in predicted_prob:
				#p[0] *= 0.85
			#print(predicted_prob)
			predictions = predicted_prob.argmax(axis=-1)
			# get most common emotion
			counts = np.bincount(predictions)
			most_common = counts.argmax()
			# get average conf level of most common emotion
			avg_conf = 0
			for prob in predicted_prob:
				avg_conf += prob[most_common]
			avg_conf /= float(predicted_prob.shape[0])
			conf_level = round(avg_conf * 100, 2)
			# update result
			result = "8:0%:"
			if conf_level > 20:
				result = str(most_common) + ":" + str(conf_level) + "%:"
			resultQueue.put((timestamp, result))
			print("prediction result: " + emotion[most_common] + ", time lag = " + str(int(getStrTime()) - int(timestamp)))
		else:
			faceMutex.release()
	print("predictImg thread ending")

	
def checkNLPResult(NLPQueue):
	print("NLP empty")
	while not NLPQueue.empty():
		timeNLP = NLPQueue.get()
		print("Got NLP response: " + timeNLP[1] + "at time " + timeNLP[0])
		

def predictLie():
	pass

#if __name__ == '__main__':
def start_server():
	print("server starting")

	#newInfo = False
	endSend = False
	stringData = None
	imgQueue = Queue()
	imgMutex = Lock()
	faceQueue = Queue()
	faceMutex = Lock()
	resultQueue = Queue()
	#NLPQueue = Queue()
	end = Value('i', 0)
	framesTaken = Value('i', 0)
	framesProcessed = Value('i', 0)
	frame_num = 10


	#predictor_path = 'shape_predictor_68_face_landmarks.dat'
	#detector = dlib.get_frontal_face_detector()
	#predictor = dlib.shape_predictor(predictor_path)

	workerProcesses = []
	for i in range(int(frame_num/2)):
		workerProcesses.append(Process(target=assignJob, args=(imgQueue, resultQueue, imgMutex, faceQueue, end, framesProcessed, frame_num, i)))
		workerProcesses[i].start()
		print("started workers")
	
	predictProcesses = []
	for i in range(5):
		predictProcesses.append(Process(target=predictImg, args=(faceQueue, faceMutex, resultQueue, frame_num, end)))
		predictProcesses[i].start()
		print("started predict image process")
	
	NLPProcess = Process(target=NLP.startNLP, args=(NLPQueue, end))
	NLPProcess.start()
	print("started predicting speech")
	
	# NLPProcess = Process(target=predictSpeech)
	# predictProcess.start()
	# print("started predict speech process")
	
	# Start a socket listening for connections on 0.0.0.0:8000
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind(('0.0.0.0', 8000))
	server_socket.listen(0)
	
	# Accept a single connection and make a file-like object out of it
	recv_connection = server_socket.accept()[0].makefile('rb')
	image_stream = io.BytesIO()
	
	# Start a client socket
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client_socket.connect(('10.19.22.17', 8001))
	print("client connected!")
	clientThread = threading.Thread(target=send_process, args=(client_socket, endSend, resultQueue))
	clientThread.start()
	
	faces = []
	frame_count = 0
	count = 0
	imgcounter = 0
	
	try:
		while True:
			# Read the length of the image as a 32-bit unsigned int. If the
			# length is zero, quit the loop
			image_len = struct.unpack('<L', recv_connection.read(struct.calcsize('<L')))[0]
			if image_len == 0:
				# tell client to end
				print("rec'd 0 size end string")
				endSend = True
				end.value = 1
				break
			# Construct a stream to hold the image data and read the image
			# data from the recv_connection
			
			#print("receiving data!")
			image_stream.write(recv_connection.read(image_len))
			#print("received frame #" + str(count) + " at time = " + str(time.time()))
			count += 1
			# Rewind the stream, open it as an image with PIL and do some
			# processing on it
			image_stream.seek(0)
			
			data = np.fromstring(image_stream.getvalue(), dtype=np.uint8)
			img = cv2.imdecode(data, 1)
			cv2.imshow("image", img)
			
			#img = np.asarray(Image.open(image_stream))
			
			#print("saving image")
			#savedImage = Image.fromarray(img)
			#img.save("my_image" + str(imgcounter) + ".jpg")
			#imgcounter += 1
			#print("image saved")
			
			imgQueue.put((getStrTime(), img))
			
			global currentFrame
			#currentFrame = img[:,:,::-1]
			currentFrame = img[:,:,:]
	
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
			
			#FOR TESTING
			#checkNLPResult(NLPQueue)
			#FOR TESTING
			
			
			if cv2.waitKey(5) & 0xFF == ord('q'):
				break
				
	finally:
		for worker in workerProcesses:
			worker.join()
		for worker in predictProcesses:
			worker.join()
		NLPProcess.join()
		clientThread.join()
		recv_connection.close()
		server_socket.close()
		cv2.destroyAllWindows()
		
		
if __name__ == '__main__':
	start_server()