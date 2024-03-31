import tensorflow as tf
import numpy as np

def custom_init(shape, dtype):
    size = tf.reduce_prod(shape)
    range = tf.range(size, dtype = dtype)
    return tf.reshape(range, shape)

def test_cnn_1_channel():
    input = tf.keras.layers.Input((10, 10, 1))
    cnn_lay =  tf.keras.layers.Conv2D(10, 2, kernel_initializer = custom_init)
    output = cnn_lay(input)
    model = tf.keras.Model(inputs = input, outputs = output)
    strange_img = np.arange(100).reshape(1, 10, 10, 1)
    print("weights:\n", tf.transpose(cnn_lay.weights[0], perm = (3, 2, 0, 1)))
    print("image:\n", strange_img.squeeze())
    print("output:\n", tf.transpose(tf.squeeze(model(strange_img)), (2, 0, 1)))

def test_cnn_3_channel():
    input = tf.keras.layers.Input((10, 10, 3))
    cnn_lay =  tf.keras.layers.Conv2D(10, 2, kernel_initializer = custom_init)
    output = cnn_lay(input)
    model = tf.keras.Model(inputs = input, outputs = output)
    strange_img = np.arange(300).reshape(1, 10, 10, 3)
    print("weights:\n", tf.transpose(cnn_lay.weights[0], perm = (3, 2, 0, 1)))
    print("image:\n", strange_img.squeeze())
    print("output:\n", tf.transpose(tf.squeeze(model(strange_img)), (2, 0, 1)))

if __name__ == "__main__":
    #test_cnn_1_channel()
    test_cnn_3_channel()
