import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

img_rows, img_cols = 28, 28 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

MODEL_FILENAME = "MNIST.hdf5"

model = load_model(MODEL_FILENAME)

image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.show()
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())