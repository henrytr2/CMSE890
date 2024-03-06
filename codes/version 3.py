import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# A) Prepare the Image Dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = datagen.flow_from_directory(
    'D:/CODES/TransClassImgData',
    target_size=(224, 256),
    batch_size=8,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'D:/CODES/TransClassImgData',
    target_size=(224, 256),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

# B) Build the Network Architecture
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 256, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(1024, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(1024, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
model.add(layers.Dense(1, activation='sigmoid'))
# C) Train the built convolutional neural network
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# D) Network assessment - Testing the network
test_generator = datagen.flow_from_directory(
    'D:/CODES/TransClassImgData',
    target_size=(224, 256),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

Y_pred = model.predict(test_generator)
Y_testing = test_generator.classes

# Convert probabilities to class labels
Y_pred_binary = (Y_pred > 0.5).astype(int)

# Calculate accuracy and confusion matrix
accuracy = np.mean(Y_pred_binary.flatten() == Y_testing)
confusion_matrix = tf.math.confusion_matrix(Y_testing, Y_pred_binary.flatten())

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
