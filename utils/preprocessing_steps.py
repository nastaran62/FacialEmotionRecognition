import cv2
import dlib
import time
import numpy as np
from imutils import face_utils
from imutils.face_utils import FaceAligner

SHAPE_PREDICTOR = "utils/shape_predictor_68_face_landmarks.dat"

def face_detection(image):
    '''
    Detect face in an image
    @param image
    @type: numpy.ndarray

    @rtype: numpy.ndarray
    @return: face image
    @note: return none if cannot detect face or detect multiple face
    '''
    # initialize dlib's face detector (HOG-based)
    detector = dlib.get_frontal_face_detector()

    rects = detector(image, 1)
    if len(rects) != 1:
        return None
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    X = 0 if x<0 else x
    Y = 0 if y<0 else y
    return image[Y:y+h, X:x+w]

def detect_and_align_face(image):
    (width, height) = image.shape

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 1)
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    if len(rects) != 1:
        return None
    fa = FaceAligner(predictor, desiredFaceWidth=width)
    face_aligned = fa.align(image, image, rects[0])

    rects_aligned = detector(face_aligned, 1)
    if len(rects_aligned) != 1:
        return None
    (x, y, w, h) = face_utils.rect_to_bb(rects_aligned[0])
    X = 0 if x<0 else x
    Y = 0 if y<0 else y
    output = face_aligned[Y:y+h, X:x+w]
    return output

def normalization(image, width, height):
    img = cv2.resize(image, (width, height))
    normalizedImg = np.zeros((width, height))
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    return normalizedImg


if __name__ == "__main__":
    img = "/home/nastaran/Documents/Projects/git-projects/FacialEmotionRecognition/dataset/training/anger/6948.jpg"
    #img = ("/home/nastaran/Documents/Projects/git-projects/created_files/images/img_1_2683_27_.jpg")
    gray_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    image = detect_and_align_face(gray_image)
    print(image.shape)
    print(image)
    print(image.shape)
    output_image = normalization(image, 48, 48)
    print(output_image.shape)

    cv2.imshow('dst_rt', output_image)
    cv2.imshow('orig', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
