import cv2
import os
from PIL import Image
import imageio  # TODO: install this to save gif

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

sampleNum = 0
person = "G005"
path = os.getcwd()

# make person folder
path_person = os.path.join(path, person)
if not os.path.exists(path_person):
    os.mkdir(path_person)

# make emotion folder
# change emotion string according to...
# second digit as your face facing
#      0: front 1: up 2: down
# last digit as emotion:
#      0: neutral, 1:anger, 2:contempt, 3:disgust, 4:fear, 5:happy, 6:sadness, 7:surprise
direction = ["front", "up", "down"]
hint = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
emotion = "027"  # TODO: change emotion number here

print("Please face " + direction[int(emotion[1])] + " and show " + hint[int(emotion[-1])])
print()

path_emotion_data = os.path.join(path_person, emotion)
if not os.path.exists(path_emotion_data):
    os.mkdir(path_emotion_data)

# record the label according to CK+ format
path_label = os.path.join(path, "labels", person, emotion)
if not os.path.exists(os.path.join(path, "labels")):
    os.mkdir(os.path.join(path, "labels"))
if not os.path.exists(os.path.join(path, "labels", person)):
    os.mkdir(os.path.join(path, "labels", person))
if not os.path.exists(path_label):
    os.mkdir(path_label)
file = open(os.path.join(path_label, "emotion.txt"), "w")
file.write(emotion[-1])
file.close()

# save images to gif
images = []


# to preview our sequence with rectangle as a gif
def save_gif():
    gif_name = person + "_" + direction[int(emotion[1])] + "_" + hint[int(emotion[-1])] + ".gif"
    gif_path = os.path.join(path_person, gif_name)
    imageio.mimsave(gif_path, images)
    print("Saved " + gif_name)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame')
    print("Recording...")

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # make sure our faces are detected before saving
        faces = face_cascade.detectMultiScale(rgb, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1
            if w > h:
                s = w
            else:
                s = h
            cv2.rectangle(rgb, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2)

            save_path = os.path.join(path_emotion_data, person + "_" + emotion + "_" + str(sampleNum) + '.jpg')
            print("Saved",  save_path)
            images.append(rgb)
            # saving the frame without rectangle
            cv2.imwrite(save_path, frame)
            cv2.waitKey(100)

        cv2.imshow('frame', rgb)  # see your face with rectangle

        if sampleNum > 9:  # take 10 frames, maybe use 6 out of them, please do not manually delete frames~
            break
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print()
    save_gif()
