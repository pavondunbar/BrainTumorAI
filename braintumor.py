# Brain Tumor AI Model
# Developed by Pavon Dunbar, August 15, 2023

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from PIL import Image

# Define the number of classes
num_classes = 4  # Replace with the actual number of classes in your dataset

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    './training',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    './training',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define class names based on your dataset
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Model building
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Update the number of classes here
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model weights
model.save_weights('model_weights.h5')

# Model evaluation
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')

# Additional evaluation metrics
validation_generator.reset()  # Reset generator to start from the beginning
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute precision, recall, and F1-score
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

# Inference
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

# Submit Test Brain MRI Image For Analysis
image_path = './brain-mri-m-1.jpeg'
predicted_class = predict_tumor(image_path, class_names)  # Provide class_names argument
print(f'Predicted class: {predicted_class}')

