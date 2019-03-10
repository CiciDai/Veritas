path = "/CKP_condensed"  # path to your condensed ck+ folder
save_path = "/npy"

image_path = os.path.join(path, 'cohn-kanade-images')
label_path = os.path.join(path, 'Emotion_labels')

print '\nSaving CK+ images and labels to .npy files.'

# Get number of images
types = ('*.png', '*.jpg')
all_image_paths = []
for files in types:
    all_image_paths.extend(glob.glob(os.path.join(image_path, '*/*/', files)))
all_img = sorted(all_image_paths)
for a in all_img:
    print(a)
num_samples = len(all_img)
print(num_samples)

X, y= make_data_label_mats(image_path, label_path, num_samples)

def make_data_label_mats(all_images_path, all_labels_path, num_samples):
    # Initialize the data of interest
    image_shape = (96, 96, 1)
    X = numpy.zeros((num_samples, image_shape[0],
                     image_shape[1], image_shape[2]), dtype='float32')
    y = numpy.zeros((num_samples), dtype='int64')

    total_sample_count = 0
    subj_list = sorted(os.listdir(all_images_path))

    # For each subject folder:
    for i, subj in enumerate(subj_list):
        seq_path = os.path.join(all_images_path, subj)
        if subj == '.DS_Store':
            os.remove(seq_path)
            continue

        print 'Subject: %d - %s' % (i, subj)

        # For each individual sequence in the subject folder:
        seq_list = sorted(os.listdir(seq_path))
        for j, seq in enumerate(seq_list):
            # Get the images of the sequence and the emotion label
            if seq == '.DS_Store':
                os.remove(os.path.join(all_images_path, subj, seq))
                continue

            images = read_images(all_images_path, subj, seq,
                                      image_shape)
            if images is not None:
                label = read_label(all_labels_path, subj, seq)
                label_vec = numpy.array([0, label, label, label, label,
                                         label, label, label, label, label])
                index_slice = slice(total_sample_count,
                                    total_sample_count+len(images))
                print("slice", index_slice, "num samples", num_samples)
                X[index_slice] = images
                if y[index_slice].shape < label_vec.shape:
                    y[index_slice] = label_vec[1:y[index_slice].shape[0]+1]
                else:
                    y[index_slice] = label_vec
                total_sample_count += len(images)

    return X, y

def read_images(all_images_path, subj, seq, image_shape):
    image_file_path = os.path.join(all_images_path, subj, seq)
    image_files = sorted(os.listdir(image_file_path))

    for i, image_file in enumerate(image_files):
        if image_file == '_DS_Store':
            os.remove(os.path.join(image_file_path, image_file))

    image_files = sorted(os.listdir(image_file_path))
    num_images = len(image_files)
    images = numpy.zeros((num_images, image_shape[0],
                          image_shape[1], image_shape[2]))

    for i, image_file in enumerate(image_files):
        if image_file == '_DS_Store':
            os.remove(os.path.join(image_file_path, image_file))
            continue

        # print image_file
        print(image_file)
        current_file = os.path.join(image_file_path, image_file)
        I = cv2.imread(current_file)
        if I.shape[:2] > (96, 96):
            print("Resize image larger than 96x96")
            I = cv2.resize(I, (96, 96))
        if I.ndim == 3:
            I = I[:, :, 0]
            I = I[:, :, numpy.newaxis]
        images[i, :, :, :] = I

    return images

def read_label(all_labels_path, subj, seq):
    label_file_path = os.path.join(all_labels_path, subj, seq)
    label_file = os.listdir(label_file_path)[0]
    f = open(os.path.join(label_file_path, label_file))
    label = f.read()
    f.close()
    # print label
    label = int(float(label))

    return label

def save_out_data(path, X, y):
    if not os.path.exists(path):
        os.makedirs(path)
    numpy.savez_compressed(os.path.join(path, 'XY96_34.npz'), x=X, y=y)
