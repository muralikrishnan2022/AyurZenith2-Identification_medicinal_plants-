# Install necessary packages
!pip install opencv-python-headless
!pip install ipywidgets
!pip install pyvirtualdisplay
!pip install easyprocess
!apt-get install -y xvfb

# Import libraries
import cv2
import numpy as np
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a function to process captured frames
def process_frame(data_url, model, data):
    binary_image = b64decode(data_url.split(',')[1])
    nparr = np.frombuffer(binary_image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the frame for classification
    resized_frame = cv2.resize(frame, (128, 128))
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_input = frame_normalized[np.newaxis, ...]

    # Perform inference with the model
    predictions = model.predict(frame_input)
    predicted_class = np.argmax(predictions[0])
    class_name = data.class_names[predicted_class]

    # Display the frame with classification result
    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    display(frame)

# Define a function for capturing frames
def capture(data, model, data_dir):
    process_frame(data, model, data)

# Register the capture function with the notebook
from google.colab import output
output.register_callback('notebook.capture', capture)

# JavaScript code for camera access
javascript = """
async function startCamera() {
   const div = document.createElement('div');
   document.body.appendChild(div);

   const video = document.createElement('video');
   video.style.display = 'block';
   const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });

   div.appendChild(video);
   video.srcObject = stream;

   await video.play();

   // Resize the output to fit the output div
   google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

   // Define a function to capture a frame from the webcam
   function captureFrame() {
       const canvas = document.createElement('canvas');
       canvas.width = video.videoWidth;
       canvas.height = video.videoHeight;
       canvas.getContext('2d').drawImage(video, 0, 0);
       return canvas.toDataURL('image/jpeg', 0.8);
   }

   // Provide a button to capture a frame
   const btn = document.createElement('button');
   btn.textContent = 'Capture';
   div.appendChild(btn);

   btn.onclick = () => {
       const frame = captureFrame();
       google.colab.kernel.invokeFunction('notebook.capture', [frame], {});
   };
}

startCamera();
"""
display(Javascript(javascript))

# Load the dataset
data = tf.keras.utils.image_dataset_from_directory(
    '/content/Training_dataset11',
    image_size=(128, 128),
    batch_size=16,
    validation_split=0.2,
    subset="training",
    seed=42
)

# Data normalization function
def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

# Create an ImageDataGenerator for data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the ResNet model without pre-trained weights
resnet_model = keras.applications.ResNet50(
    include_top=False,
    input_shape=(128, 128, 3),
    weights='imagenet'  # Use pre-trained weights
)

# Adjust the learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Customize the top layers for your number of output classes
num_classes = len(data.class_names)
top_layers = keras.Sequential([
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),      #Additional layers of dense necessary acc to dataset
    layers.Dropout(0.5),                       #Should have added more to avoid overfitting
    layers.Dense(num_classes, activation='softmax')
])

# Combine the base ResNet model and the custom top layers
model = keras.Sequential([
    resnet_model,
    top_layers
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with more epochs
history = model.fit(
    data,
    epochs=5,  # Increase the number of epochs
    validation_data=(data)
)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Create a loop for webcam integration
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the webcam frame
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Classify the frame
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions[0])
    class_name = data.class_names[predicted_class]

    # Display the frame with classification result
    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    display(frame)
