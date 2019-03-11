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
model = tf.keras.models.load_model("all_model.h5")
print("Loaded model from disk")

# compile loaded model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.train.GradientDescentOptimizer(0.01),
              metrics=['accuracy'])

# evaluate the loaded model
# path = "../npy/npy_files/"
# zip = np.load(os.path.join(path, "XY96_2D_34.npz"))
# x = zip['x']
# y = zip['y']
# model.evaluate(x, y)

# load the labels for emotion
emotion = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

currentFrame = None
circBuffer = np.zeros((30,), dtype=int)

def assignJob(imgQueue, resultQueue, imgMutex, faceQueue, end, framesProcessed, frame_num):
    #while endSend is False or imgQueue.empty() is False:
    while end.value == 0 or imgQueue.empty() is False:
    #if imgQueue.empty() is False:
        
        imgMutex.acquire()
        if imgQueue.empty() is True:
            imgMutex.release()
        else:
            img = imgQueue.get()
            imgMutex.release()
            face = processImg(img)
            if face is not None:
                faceQueue.put(face)
            framesProcessed.value += 1
            if faceQueue.qsize() >= frame_num:
                continue
                

def processImg(img):
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
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            #gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            #face[person] = cv2.resize(gray, (96, 96))
            face = cv2.resize(gray, (96, 96))
            
        # reshape face numpy into the shape of trained data
        face = face[:, :, np.newaxis]
        
        return face
    return None
    
def predictImg(faceQueue, resultQueue, framesProcessed, frame_num, end):
    while end.value == 0:
        if faceQueue.qsize() >= frame_num:
            faces = np.zeros((frame_num, 96, 96, 1))
            for i in range(frame_num):
                #faces.append(faceQueue.get())
                faces[i] = faceQueue.get()
                
            if len(faces) == 0:
                continue
            
            #faces = np.array(faces)
            
            predicted_prob = model.predict(faces)  # predict
            print("predicted_prob:")
            print(predicted_prob)
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
            result = "8::"
            if conf_level > 20:
                result = str(most_common) + ":" + str(conf_level) + "%:"
            resultQueue.put(result)
            print("prediction result: " + emotion[most_common])
	
	
if __name__ == "__main__":

    endSend = False
    stringData = None
    imgQueue = Queue()
    imgMutex = Lock()
    faceQueue = Queue()
    resultQueue = Queue()
    end = Value('i', 0)
    framesProcessed = Value('i', 0)
    frame_num = 6

    
    workerProcesses = []
    for i in range(frame_num):
        workerProcesses.append(Process(target=assignJob, args=(imgQueue, resultQueue, imgMutex, faceQueue, end, framesProcessed, frame_num)))
        workerProcesses[i].start()
        print("started workers")
        
    predictProcess = Process(target=predictImg, args=(faceQueue, resultQueue, framesProcessed, frame_num, end))
    predictProcess.start()
    print("started predict process")
    
    
    faces = []
    frame_count = 0
    count = 0
    imgcounter = 0
    
    time.sleep(10)
    
    cap = cv2.VideoCapture("myvideo.mjpeg")
    cv2.namedWindow('frame')
    
    try:
        while True:
            # 1) capture a new frame from the camera
            t1 = cv2.getTickCount()
            ret, frame = cap.read()
            
            cv2.imshow("frame", frame)
            
            imgQueue.put(frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    finally:
        for worker in workerProcesses:
            worker.join()
        predictProcess.join()
        cv2.destroyAllWindows()