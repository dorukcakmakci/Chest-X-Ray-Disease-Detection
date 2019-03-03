from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras import regularizers, optimizers
from keras.utils import plot_model
import cv2

# dimensions of our images.
img_width, img_height = 224, 224

# uncomment below when the data is ready at the directed folders --
# train_data_dir = "train/"
# validation_data_dir = "val/"
# test_data_dir = "test/"

nb_train_samples = 5237
nb_validation_samples = 1747
epochs = 100
batch_size = 64

if K.image_data_format() == "channels_first":
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()

model.add(Conv2D(32, (7,7), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))

model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(4))
model.add(Activation('softmax'))

# Uncomment to load previous weights
# model.load_weights("previous_weights.h5")

sgd = optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=False)

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1. / 255, zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode = "categorical",
    color_mode = "grayscale")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode = "grayscale")

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode = "categorical",
    color_mode = "grayscale")


model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

print(train_generator.class_indices)

prediction = model.evaluate_generator(test_generator, verbose=1)
print(model.metrics_names)
print("Loss: " + str(prediction[0]))
print("Accuracy " + str(prediction[1]))

# plot_model(model, to_file='model.png')
















