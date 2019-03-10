import os
import pandas as pd
import numpy as np
import cv2

# load the excel sheet
path = "/CASME2"
# file_name = "CASME2-coding-20140508.xlsx"
file_name = "CASME2-coding-2019217.xlsx"
sheet = "Sheet1"
file_path = os.path.join(path, file_name)
df = pd.read_excel(io=file_path, sheet_name=sheet)
# print(df.head(5))

# load list for subjection
subj = df['Subject'].values
# print(subj)
# load list for sequences
seq = df["Filename"].values
# print(seq)
# load list for onset frame
onset = df["OnsetFrame"].values
# load list for apex frame
apex = df['ApexFrame'].values
# print(apex)
# load list for emotion labels
emotion_label = df['Estimated Emotion'].values
# make emotion labels into list of int
emotion = ['others', 'anger', 'repression', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
label = np.zeros(emotion_label.shape, dtype='int64')
for i, e in enumerate(emotion_label):
    label[i] = emotion.index(e)
# print(label)

def create():
    img_path = os.path.join(path, "CASME2_RAW_selected")
    save_path ="/models/CASME2"
    # get 8 images around the apex as well as the apex image, and first image,
    # total = 10 each sequence
    num_samples = len(seq) * 10

    X, y = make_data_label_mats(img_path, num_samples)
    print(X.shape, y.shape)
    save_out_data(save_path, X, y)

def make_data_label_mats(img_path, num_samples):
    image_shape = [96, 96, 1]

    X = np.zeros((num_samples, image_shape[0],
                  image_shape[1], image_shape[2]), dtype='float32')
    y = np.zeros(num_samples, dtype='int64')

    total_sample_count = 0
    count_folder = 0

    subj_list = sorted(os.listdir(img_path))
    for i, subject in enumerate(subj_list):
        seq_path = os.path.join(img_path, subject)
        if subject == '.DS_Store':
            os.remove(seq_path)
            continue
        print ('Subject:', subject)

        # For each individual sequence in the subject folder:
        seq_list = sorted(os.listdir(seq_path))
        for j, sequence in enumerate(seq_list):
            # Get the images of the sequence and the emotion label
            if sequence == '.DS_Store':
                os.remove(os.path.join(seq_path, sequence))
                continue
            print ('\t Sequence:', sequence)

            images = read_images(img_path, subject, sequence, image_shape, count_folder)
            cur_label = label[count_folder]
            label_vec = np.array(0)  # first one is neutral
            label_vec = np.append(label_vec, np.full((len(images)-1), cur_label))  # add the rest
            index_slice = slice(total_sample_count,
                                total_sample_count + len(images))
            # print("slice", index_slice, "num samples", num_samples)
            X[index_slice] = images
            y[index_slice] = label_vec
            total_sample_count += len(images)
            count_folder += 1

    return X, y

def read_images(img_path, subject, sequence, image_shape, count_folder):
    image_file_path = os.path.join(img_path, subject, sequence)
    image_files = sorted(os.listdir(image_file_path))

    num_images = 10
    images = np.zeros((num_images, image_shape[0],
                        image_shape[1], image_shape[2]))

    count = 0
    for image_file in image_files:
        # save image file
        if image_file == ".DS_Store":
            print("removing .DS_Store")
            os.remove(os.path.join(image_file_path, image_file))
            continue
        if image_file != '_DS_Store' and is_near_apex(image_file, count_folder):
            # print(image_file)
            current_file = os.path.join(image_file_path, image_file)
            img = cv2.imread(current_file)
            # find face
            img = face_detector(img)
            # reduce image size to gray and 96x96
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (96, 96))
            img = img[:, :, np.newaxis]
            images[count, :, :, :] = img
            count += 1
    # return the 10 images around apex
    return images

def face_detector(I, scale_factor=1.3, min_neighbors=5,
                  min_size_scalar=0.25, max_size_scalar=0.75):
    module_path = "../data collection/"
    classifier_path = os.path.join(module_path,
                                   'haarcascade_frontalface_default.xml')
    detector = cv2.CascadeClassifier(classifier_path)

    height, width, num_channels = I.shape
    min_dim = np.min([height, width])
    min_size = (int(min_dim * min_size_scalar),
                int(min_dim * min_size_scalar))
    max_size = (int(min_dim * max_size_scalar),
                int(min_dim * max_size_scalar))

    faces = detector.detectMultiScale(I, scale_factor,
                                      min_neighbors, 0,
                                      min_size, max_size)
    if len(faces) > 0:
        loc = faces[0]
        (x, y, w, h) = loc
        i_crop = I[y:y + h, x:x + w, :]
        return i_crop
    else:
        print('can''t find face, try again')
        i_crop = face_detector(I, scale_factor=1.05,
                               min_neighbors=1,
                               min_size_scalar=0.1,
                               max_size_scalar=0.9)
        return i_crop

def is_near_apex(img_name, count_folder):
    num = img_name.split("img")
    num = num[1]
    num = num.split(".")
    num = int(num[0])
    # print("num", num, "apex", apex[count_folder])
    first_num = onset[count_folder]
    apex_num = apex[count_folder]
    if type(apex_num) == str:
        # special case when subj 4's EP12_01f sequence has no apex
        apex_num = 261 + ((321-261)//2)
    if num == first_num or abs(apex_num - num) <= 4:
        # print("include", num)
        return True
    else:
        return False

def save_out_data(save_path, X, y):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    npz_name = input("Give compressed file a name: ")
    npz_name = "XY_Casme2_" + npz_name +".npz"
    np.savez_compressed(os.path.join(save_path, npz_name), x=X, y=y)
    print(X.shape[0], "data saved to", npz_name, "!")

create()
