import time
import pandas
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from model import conv_model

img_width, img_height = 48, 48
BATCH_SIZE = 32
CLASS_MODE = "sparse"
training_images_path = "preprocessed_data/training"
validation_images_path = "preprocessed_data/validation"
training_csv_path = "csv/training.csv"
validation_csv_path = "csv/validation.csv"
EPOCHS = 100

def training():
    model = conv_model()
        # Save the model according to the conditions

    #Adding custom Layers
    print("data augmentation")
    train_data_generator = \
        ImageDataGenerator(rescale=1./255,
                           rotation_range=20,
                           zoom_range=0.15,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.15,
                           horizontal_flip=True,
                           fill_mode="nearest")
    train_data_frame = pandas.read_csv(training_csv_path, dtype=str)
    train_generator = \
        train_data_generator.flow_from_dataframe(dataframe=train_data_frame,
                                                 directory=training_images_path,
                                                 validate_filenames=True,
                                                 x_col="file_name",
                                                 y_col="emotion",
                                                 target_size=(img_width, img_height),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode=CLASS_MODE,
                                                 shuffle=True,
                                                 color_mode="grayscale")

    validation_data_generator = \
        ImageDataGenerator(rescale=1./255)
    validation_data_frame = pandas.read_csv(validation_csv_path, dtype=str)
    validation_generator = \
        validation_data_generator.flow_from_dataframe(dataframe=validation_data_frame,
                                                      directory=validation_images_path,
                                                      validate_filenames=True,
                                                      x_col="file_name",
                                                      y_col="emotion",
                                                      target_size=(img_width, img_height),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=CLASS_MODE,
                                                      shuffle=True,
                                                      color_mode="grayscale")



    checkpoint = ModelCheckpoint('models/model-{}.h5'.format(int(time.time())), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir="logs/{}".format("model-{}".format(int(time.time()))))

    train_data_count, col_count = train_data_frame.shape
    validation_data_count, col_count = validation_data_frame.shape
    model.fit_generator(train_generator,
                        steps_per_epoch=train_data_count // BATCH_SIZE,
                        validation_steps=validation_data_count // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=validation_generator,
                        callbacks=[tensorboard, early, checkpoint],
                        )
    model.save('models/model-{}.model'.format(int(time.time())))

training()
