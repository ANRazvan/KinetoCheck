import pandas as pd
import numpy as np
import os


# SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.txt
# SubjectID: id uniquely identifying the person performing the exercise
# DateID: id identifying the session in which the person was recorded
# GestureLabel: Label identifying the gesture; possible values are from 0 to 8
# RepetitionNumber: Each gesture was repeated several times and this shows the repetition number
# CorrectLabel: A value of 1 represents a gesture labeled as being correctly executed, while a value of 2 is for a gesture labeled as incorrect
# Position: Some of the persons performed the gestures sitting on a chair or wheelchair, while others standing
from sklearn.utils import resample


def read_data(dataset_path):
    all_files = os.listdir(dataset_path)

    x_data = []
    labels = []
    date_ids = []
    subject_ids = []
    exercise_ids = []
    positions = []
    repetitions = []

    index = -1

    for filename in all_files:
        index += 1
        if 'txt' not in filename:
            continue

        subject_id, date_id, exercise_id, repetition, label, position = extract_info_from_filename(filename)
        # the labels = 3 are poorly executed and are not considered as part of the dataset
        if label == 2:
            continue

        current_exercise = read_txt_file(dataset_path + filename)
        # if index == 0:
        #     print(filename)
        #     print(current_exercise)

        x_data.append(current_exercise)
        subject_ids.append(subject_id)
        exercise_ids.append(exercise_id)
        date_ids.append(date_id)
        repetitions.append(repetition)
        positions.append(position)
        labels.append(label)

    labels_numpy = np.array(labels)
    return x_data, labels

def downsampling(x_data, labels):
    x_data = np.array(x_data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print("Counts: ", class_counts)

    # Create a dictionary to store the data and labels for each class
    class_data = {c: x_data[labels == c] for c in unique_classes}
    class_labels = {c: labels[labels == c] for c in unique_classes}

    min_count = min(class_counts)
    downsampled_x = []
    downsampled_y = []

    for c in unique_classes:
        x_class, y_class = resample(class_data[c], class_labels[c], replace=False, n_samples=min_count, random_state=42)
        downsampled_x.extend(x_class)
        downsampled_y.extend(y_class)

    downsampled_y_numpy = np.array(downsampled_y, dtype=np.float32)

    unique_classes, class_counts = np.unique(downsampled_y_numpy, return_counts=True)
    print("Label counts after downsampling: ", class_counts)

    return downsampled_x, downsampled_y

def extract_info_from_filename(filename):
    '''
       split the filename and return metadata:
           - subject_id
           - date_id
           - exercise_id
           - repetition_number
           - label
           - position: it can be stand, chair or wheelchair
    '''

    pieces = filename.split("_")

    subject_id = pieces[0]
    date_id = pieces[1]
    exercise_id = pieces[2]
    repetition_number = pieces[3]
    label = int(pieces[4]) - 1

    position_char = pieces[5].replace('.txt', '')

    if position_char.lower() in ("stand", "stand-frame"):
        position = 1
    elif position_char.lower() in ("chair", "sit"):
        position = 2
    else:
        position = 3  # wheelchair or other

    return subject_id, date_id, exercise_id, repetition_number, label, position


def read_txt_file(filename):
    # Skip "Version0.1" header lines that some files contain
    with open(filename, 'r') as f:
        lines_to_skip = 0
        for line in f:
            if line.strip() == "Version0.1":
                lines_to_skip += 1
            else:
                break

    df = pd.read_csv(filename, sep=',', header=None, index_col=False, skiprows=lines_to_skip)
    return df.values
