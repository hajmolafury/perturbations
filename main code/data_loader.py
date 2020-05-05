import tensorflow as tf
import numpy as np

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train_data, y_train_cold),(x_test_data,y_test_cold) = mnist.load_data()
    x_train_data, x_test_data = x_train_data / 255.0, x_test_data / 255.0


    with tf.Session() as sess:
        y_train = sess.run(tf.one_hot(y_train_cold,10))
        y_test =  sess.run(tf.one_hot(y_test_cold,10))


    x_train=[]
    x_test=[]

    n_train = len(x_train_data)
    n_test = len(x_test_data)

    for i in range(len(x_train_data)):
        x_train.append(np.ndarray.flatten(x_train_data[i]))
        
    for i in range(len(x_test_data)):
        x_test.append(np.ndarray.flatten(x_test_data[i]))


    x_train = np.reshape(x_train,[n_train,784])
    x_test = np.reshape(x_test,[n_test, 784])

    return (x_train, y_train),(x_test, y_test)