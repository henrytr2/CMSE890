import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set the path to your dataset
base_dir = 'D:/CODES'
data_dir = os.path.join(base_dir, 'TransClassImgData')

# Load and read the image folder as "imds" and its 2 subfolders along with the 2 associated labels of each image
# Use ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create training, validation, and testing generators
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 256),
    batch_size=8,
    class_mode='binary',  # 'binary' for binary classification
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 256),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

# Build the CNN architecture
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 256, 3)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, strides=2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, strides=2),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, strides=2),
    
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, strides=2),
    
    layers.Conv2D(1024, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(1024, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2, strides=2),
    
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')  # 'sigmoid' for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1
)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Network assessment - Testing the network
test_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 256),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
