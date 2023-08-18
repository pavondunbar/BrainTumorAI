# Predict Tumor Python Script
# Developed by Pavon Dunbar, August 15, 2023

import tensorflow as tf
import numpy as np
from PIL import Image

# Define the class names based on your dataset
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load the trained model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # Update the number of classes here
])

# Load the trained model weights
model.load_weights('./model_weights.h5')  # Replace with the actual path

# Inference function
def predict_tumor(image_path, class_names):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]
    class_label = class_names[class_index]
    return class_label

# Submit Brain MRI Image for Analysis
image_path = './p-tumor.jpeg'  # Path to the new image
predicted_class = predict_tumor(image_path, class_names)
print(f'Predicted class: {predicted_class}')

