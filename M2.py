# Import library
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Flatten
from tensorflow.keras.layers import Conv2D
# Load dataset
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Normalize data
X_train = X_train.astype('float32') / 255.0
unique_labels = np.unique(y_train)
print(y_train)
# Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save model
model.save('my_model.h5')