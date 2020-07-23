# FacialEmotionRecognition

This repository uses FER2013 dataset images to train a deep model and predicting facial expression in real time.

preprocessing.py : Preprocess images (face detection) and prepare them for data generators

training.py : Training the model (model folder should be created)

predict.py  Can predict facial expression using the trained model in model folder 

(It can be done in two different ways: 1- reading data from camera and predict online. 2- predicting a video (frame bt frame) or an image facial expression) 
