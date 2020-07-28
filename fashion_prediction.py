import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Loading our dataset
data = keras.datasets.fashion_mnist

# Dividing data into training and testing data
(train_imgs, train_labels), (test_imgs, test_labels) = data.load_data()

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalizing our data
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# Creating our model
'''model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),   # relu = rectified linear unit
    keras.layers.Dense(10, activation='softmax')
    ])

# compiling our model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training our model
model.fit(train_imgs, train_labels, epochs=5)

# Saving our trained model
model.save("fashion_prediction.h5")'''

# Loading our saved model
model = keras.models.load_model("fashion_prediction.h5")

# Calculating the accuracy of our model
loss, acc = model.evaluate(test_imgs, test_labels)
print("Accuracy: ", acc)

# Prediction
predictions = model.predict(test_imgs)


for i in range(5):
    plt.grid(False)
    plt.imshow(test_imgs[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_name[test_labels[i]])
    plt.title("Predicted: " + class_name[np.argmax(predictions[i])])
    plt.show()
