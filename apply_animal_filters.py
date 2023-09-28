import cv2
import time
import numpy as np

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Define a function to create a CNN model for facial keypoints detection
def create_model():
    '''
    Define a CNN architecture where input to the model must be 96 * 96 pixel grayscale image.
    
    Returns:
    -------------
    model: A fully-connected output layer with 30 facial keypoint values
    '''

    # Initialize a sequential model
    model = models.Sequential()

    # Add a convolutional layer with 32 filters, each of size (5, 5), using 'relu' activation function
    # Input shape is set to (96, 96, 1), as it expects a grayscale image of size 96x96 pixels
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(96, 96, 1)))

    # Add a max pooling layer with a pool size of (2, 2)
    model.add(layers.MaxPooling2D((2, 2)))

    # Add another convolutional layer with 64 filters, each of size (3, 3), using 'relu' activation function
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add another max pooling layer with a pool size of (2, 2)
    model.add(layers.MaxPooling2D((2, 2)))

    # Add a dropout layer with a rate of 0.1 to reduce overfitting
    model.add(layers.Dropout(0.1))

    # Add another convolutional layer with 128 filters, each of size (3, 3), using 'relu' activation function
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Add another max pooling layer with a pool size of (2, 2)
    model.add(layers.MaxPooling2D((2, 2)))

    # Add a dropout layer with a rate of 0.2
    model.add(layers.Dropout(0.2))

    # Add another convolutional layer with 256 filters, each of size (3, 3), using 'relu' activation function
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    # Add another max pooling layer with a pool size of (2, 2)
    model.add(layers.MaxPooling2D((2, 2)))

    # Add a dropout layer with a rate of 0.3
    model.add(layers.Dropout(0.3))

    # Add another convolutional layer with 256 filters, each of size (3, 3), using 'relu' activation function
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    # Add another max pooling layer with a pool size of (2, 2)
    model.add(layers.MaxPooling2D((2, 2)))

    # Add a dropout layer with a rate of 0.3
    model.add(layers.Dropout(0.3))

    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Add fully connected layers with 'relu' activation function
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))

    # Add the output layer with 30 units (for 30 facial keypoints)
    model.add(layers.Dense(30))

    return model

# Define a function to compile the model with specific optimizer, loss, and metrics
def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Define a function to train the model
def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)

# Define a function to save the trained model
def save_model(model, fileName):
    model.save(fileName + '.h5')

# Define a function to load a previously trained model
def load_model(fileName):
    return models.load_model(fileName + '.h5')



def apply_filters(face_points, image_copy_1, image_name):
    '''
    Apply animal filters to a person's face

    Parameters:
    --------------------
    face_points: The predicted facial keypoints from the camera
    image_copy_1: Copy of original image

    Returns:
    -------------
    image_copy_1: Animals filters applied to copy of original image
    '''

    animal_filter = cv2.imread("images/"+image_name, cv2.IMREAD_UNCHANGED)

    for i in range(len(face_points)):
        # Get the width of filter depending on left and right eye brow point
        # Adjust the size of the filter slightly above eyebrow points
        filter_width = 1.1*(face_points[i][14]+15 - face_points[i][18]+15)
        scale_factor = filter_width/animal_filter.shape[1]
        sg = cv2.resize(animal_filter, None, fx=scale_factor,
                        fy=scale_factor, interpolation=cv2.INTER_AREA)

        width = sg.shape[1]
        height = sg.shape[0]

        # top left corner of animal_filter: x coordinate = average x coordinate of eyes - width/2
        # y coordinate = average y coordinate of eyes - height/2
        x1 = int((face_points[i][2]+5 + face_points[i][0]+5)/2 - width/2)
        x2 = x1 + width

        y1 = int((face_points[i][3]-65 + face_points[i][1]-65)/2 - height/3)
        y2 = y1 + height

        # Create an alpha mask based on the transparency values
        alpha_fil = np.expand_dims(sg[:, :, 3]/255.0, axis=-1)
        alpha_face = 1.0 - alpha_fil

        # Take a weighted sum of the image and the animal filter using the alpha values and (1- alpha)
        image_copy_1[y1:y2, x1:x2] = (
            alpha_fil * sg[:, :, :3] + alpha_face * image_copy_1[y1:y2, x1:x2])

    return image_copy_1


# Load the model built in the previous step
# model = load_model('models/final_model')
model = load_model('model')


# Get frontal face haar cascade
# face_cascade = cv2.CascadeClassifier(
#     'cascades/haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get webcam
camera = cv2.VideoCapture(0)

while True:
    # Read data from the webcam
    _, image = camera.read()
    image_copy = np.copy(image)
    image_copy_1 = np.copy(image)
    image_copy_2 = np.copy(image)

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Identify faces in the webcam using haar cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_keypoints = []

    # Loop through faces
    for (x, y, w, h) in faces:

        # Crop Faces
        face = gray[y:y+h, x:x+w]

        # Scale Faces to 96x96
        scaled_face = cv2.resize(face, (96, 96), 0, 0,
                                 interpolation=cv2.INTER_AREA)

        # Normalize images to be between 0 and 1
        input_image = scaled_face / 255

        # Format image to be the correct shape for the model
        input_image = np.expand_dims(input_image, axis=0)
        input_image = np.expand_dims(input_image, axis=-1)

        # Use model to predict keypoints on image
        face_points = model.predict(input_image)[0]

        # Adjust keypoints to coordinates of original image
        face_points[0::2] = face_points[0::2] * w/2 + w/2 + x
        face_points[1::2] = face_points[1::2] * h/2 + h/2 + y
        faces_keypoints.append(face_points)

        # Plot facial keypoints on image
        for point in range(15):
            cv2.circle(
                # image_copy, (face_points[2*point], face_points[2*point + 1]), 2, (255, 255, 0), -1)
                image_copy, (int(face_points[2*point]), int(face_points[2*point + 1])), 2, (255, 255, 0), -1)

        bear = apply_filters(faces_keypoints, image_copy_1, "bear.png")
        cat = apply_filters(faces_keypoints, image_copy_2, "cat.png")

        # Screen with the filter
        cv2.imshow('Screen with bear filter', bear)
        cv2.imshow('Screen with cat filter', cat)
        # Screen with facial keypoints
        cv2.imshow('Screen with facial Keypoints predicted', image_copy)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break