import cv2
import numpy as np
from utils.preprocessing_steps import face_detection, normalization
from keras.models import load_model

img_width, img_height = 48, 48

def read():
    '''
    '''
    img_array = cv2.imread('/home/nastaran/Documents/Projects/git-projects/FacialEmotionRecognition/dataset/test/disgust/1549.jpg')
    return img_array

def image_preprocessing(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = face_detection(gray_image)

    if img is None:
        return None

    # image normalization and scaling
    normalized_image = normalization(img, img_width, img_height)
    output = normalized_image.reshape(img_width, img_height, 1)
    return output

    return normalized_image

def predict():
    img = image_preprocessing(read())
    model = load_model('models/model-1570411054.h5')
    test_set = np.array([img])
    predicted_values = model.predict(test_set)
    label = np.argmax(predicted_values, axis=1)
    print("label = ", label)

emotions = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

def predict_online():
    '''
    '''
    model = load_model('models/model-1570411054.h5', compile=False)
    video_capture = cv2.VideoCapture(1)
    print("model is loaded")

    while True:
        #gray_image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        ret, frame = video_capture.read()

        img = image_preprocessing(frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break
        if img is None:
            continue

        test_set = np.array([img])
        predicted_values = model.predict(test_set/255, verbose=1)
        print(predicted_values)
        label = np.argmax(predicted_values, axis=1)
        print(label)
        print(emotions[label[0]])

    cv2.destroyAllWindows()


predict_online()
