import os
import cv2
import csv
from utils.preprocessing_steps import face_detection, normalization#, detect_and_align_face


WIDTH, HEIGHT = 48, 48

def preprocessing(data_path, preprocessing_path, name):
    '''
    Preprocess data and save the preprocessed images in a folder

    :param str image_data_path: image data path
    :note image_data_path: The image_data_path should include a folder for each emotion
           {angry, disgust, fear, happy, neutral, sad, surprise}
    '''
    image_data_path = os.path.join(data_path, name)
    emotions = os.listdir(image_data_path)
    emotions.sort()
    with open('csv/{}.csv'.format(name), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['file_name','emotion'])
        for emotion in emotions:
            class_num = emotions.index(emotion)
            path = os.path.join(image_data_path, emotion)  # create path to emotions
            for img in os.listdir(path):
                try:
                    # Convert image to gray scale, it seems that the color doesn't affect on the emotion
                    gray_image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array

                    # Put the preprocessing steps here
                    # Face detection and alignment
                    # preprocessed_image = detect_and_align_face(gray_image)

                    # Face detection (We can remove face alignment)
                    preprocessed_image = face_detection(gray_image)
                    if preprocessed_image is None:
                        continue

                    # image normalization and scaling
                    normalized_image = normalization(preprocessed_image, WIDTH, HEIGHT)
                    #writing the preprocessed data
                    file_name = "/p_{}".format(img)
                    csv_writer.writerow([file_name, class_num])
                    preprocessed_image_path = os.path.join(preprocessing_path, name)
                    cv2.imwrite(preprocessed_image_path + file_name, normalized_image)
                except Exception as e:
                    print(e)
preprocessing("dataset", "preprocessed_data", "training")
# Parameters that can be changed
#   1- using detect_and_align_face instead of face_detection
