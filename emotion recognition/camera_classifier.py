# Ying Zheng
# TECHIN 510
# 3-4-2019
# This classifier detects emotions from camera using trained classifier
# and display the detected emotions on the image

import os
import time
import cv2
import numpy as np
import tensorflow as tf

# load frontal face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load model
model = tf.keras.models.load_model("all_model.h5")
print(model.summary())
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

# initiate one sequence
frame_num = 6  # collect 10 frames to get average emotion
sequence = np.zeros((5, frame_num, 96, 96, 1))  # max five ppl for now
count_seq = [0, 0, 0, 0, 0]  # for five ppl
result = ["", "", "", "", ""]

# calculate fps
frame_rate_calc = 1
freq = cv2.getTickFrequency()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame')

    while True:
        # 1) capture a new frame from the camera
        t1 = cv2.getTickCount()
        ret, frame = cap.read()

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        top = []
        left = []
        # if there is face in frame
        if type(faces) is np.ndarray:
            face = np.zeros((faces.shape[0], 96, 96))
            person = 0
            for (x, y, w, h) in faces:
                left.append(x)
                top.append(y)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                if w > h:
                    s = w
                else:
                    s = h

                crop = frame[y:y+s, x:x+s]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                face[person] = cv2.resize(gray, (96, 96))
                person += 1

            # reshape face numpy into the shape of trained data
            face = face[:, :, :, np.newaxis]

            predictable = False
            # add frame of everyone in their sequence
            for p in range(person):
                curr_frame = count_seq[p]
                if curr_frame < frame_num:  # if this sequence is not full
                    # add frame of this person to this sequence
                    sequence[p][curr_frame] = face[p]
                    count_seq[p] += 1
                else:
                    predictable = True
                    count_seq[p] = 0

            if predictable:
                # detect whether the image contains one of the seven emotions
                # using the trained model
                exist_sequence = sequence[0: person]  # get only filled sequence
                # label everyone in frame
                for p in range(person):
                    predicted_prob = model.predict(exist_sequence[p])  # predict this person
                    predictions = predicted_prob.argmax(axis=-1)
                    # get most common emotion
                    counts = np.bincount(predictions)
                    most_common = counts.argmax()
                    # get average conf level of most common emotion
                    avg_conf = 0
                    for prob in predicted_prob:
                        avg_conf += prob[most_common]
                    avg_conf /= float(frame_num)
                    conf_level = round(avg_conf * 100, 2)
                    # update result
                    if conf_level > 20:
                        result[p] = emotion[most_common] + ": " + str(conf_level) + "%"
                    # save photo of this person's emotion
                    # name = emotion[most_common] + " " + str(time.time()) + '.png'
                    # cv2.imwrite(name, face[p])

            for p in range(person):
                cv2.putText(frame, result[p], (left[p], top[p]-14),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 250), thickness=2)
                # print(result[p])

        # calculate fps
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1 / time1
        cv2.putText(frame, str(frame_rate_calc), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 50), thickness=2)

        cv2.imshow('Facial Expression', frame)
        # This allows you to stop the script from looping by pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
