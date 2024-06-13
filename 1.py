import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import os
import pickle

# Function to load CIFAR-10 dataset batches
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

# Path to CIFAR-10 dataset directory
cifar10_dir = 'cifar-10-batches-py'

print("Loading training batches...")
# Load training batches
X_train = []
y_train = []
for i in range(1, 6):
    f = os.path.join(cifar10_dir, f'data_batch_{i}')
    X, Y = load_cifar_batch(f)
    X_train.append(X)
    y_train.append(Y)
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

print("Loading test batch...")
# Load test batch
X_test, y_test = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))

print("Normalizing images...")
# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Specify GPU usage
with tf.device('/GPU:0'):
    print("Creating CNN model...")
    # Create CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    # Compile the model
    print("Compiling the model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Callbacks for monitoring training
    callbacks = [
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f'Epoch {epoch+1}: loss = {logs["loss"]:.4f}, accuracy = {logs["accuracy"]:.4f}')
        )
    ]

    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train, batch_size=32, epochs=10, 
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)

    # Save the model
    print("Saving the model...")
    model.save('cifar10_cnn_model.h5')

print("Training completed and model saved.")
