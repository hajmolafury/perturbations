import tensorflow as tf
import numpy as np
import globalV

class tfGraph:

    def __init__(self):
        self.g = tf.Graph()

    def build_graph(self):

        with self.g.as_default():

            self.define_variables()
            error = tf.reduce_mean(np.square(self.pred - self.Y), 1)
            mse = tf.losses.mean_squared_error(predictions=self.pred, labels=self.Y)

            for i in range(globalV.n_hl + 1):
                if (i == 0):
                    self.h_star[str(i)] = tf.matmul(tf.add(self.X, self.s['in']), tf.transpose(self.w[str(i)]))
                else:
                    self.h_star[str(i)] = tf.matmul(self.x_star[str(i - 1)], tf.transpose(self.w[str(i)]))

                if (i < globalV.n_hl):
                    self.x_star[str(i)] = tf.nn.relu(tf.add(tf.add(self.h_star[str(i)], self.b[str(i)]), self.s[str(i)]))

                else:
                    pred_star = tf.nn.softmax(tf.add(tf.add(self.h_star[str(i)], self.b[str(i)]), self.s[str(i)]))

            error_star = tf.reduce_mean(np.square(pred_star - self.Y), 1)
            var = globalV.sigma ** 2
            k = -globalV.learning_rate * (error_star - error) / var

            for i in range(globalV.n_hl + 1):
                if (globalV.update_rule == 'ip'):
                    self.del_h[str(i)] = self.h_star[str(i)] - self.h[str(i)]

                elif (globalV.update_rule == 'np-hybrid'):
                    self.del_h[str(i)] = self.s[str(i)]
                    k = k * (1 - globalV.alpha)

                if (i == 0):
                    self.delta_w[str(i)] = tf.einsum('ki,kj->kij', self.del_h[str(i)], self.X)

                else:
                    self.delta_w[str(i)] = tf.einsum('ki,kj->kij', self.del_h[str(i)], self.x[str(i - 1)])

                self.delta_w[str(i)] = tf.einsum('kij,k->kij', self.delta_w[str(i)], k)
                self.delta_w[str(i)] = tf.reduce_mean(self.delta_w[str(i)], 0)
                self.delta_b[str(i)] = tf.einsum('ki,k->ki', self.del_h[str(i)], k)
                self.delta_b[str(i)] = tf.reduce_mean(self.delta_b[str(i)], 0)

                # assert tf.count_nonzero(self.delta_w[str(i)]) is not False

                self.update_w[str(i)] = tf.assign(self.w[str(i)], tf.add(self.w[str(i)], self.delta_w[str(i)]), name='update_w' + str(i))
                self.update_b[str(i)] = tf.assign(self.b[str(i)], tf.add(self.b[str(i)], self.delta_b[str(i)]), name='update_b' + str(i))

            lr_sgd=globalV.alpha * globalV.learning_rate
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_sgd, name='optimizer').minimize(mse)

            is_correct = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
            accuracy = 100 * tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.identity(accuracy, 'accuracy')


    def define_variables(self):
        self.w = {}
        self.b = {}
        self.h = {}
        self.x = {}
        self.s = {}
        self.h_star = {}
        self.x_star = {}
        self.del_h = {}
        self.delta_w = {}
        self.delta_b = {}

        self.update_w = {}
        self.update_b = {}


        xavier = tf.contrib.layers.xavier_initializer(seed=globalV.seed)
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')
        self.lr_sgd = tf.Variable(xavier([1]), dtype=tf.float32, name='lr_sgd')

        self.s['in'] = tf.zeros([1], dtype=tf.float32, name='s_in')

        for i in range(globalV.n_hl + 1):
            self.s[str(i)] = tf.zeros([1], dtype=tf.float32, name='s_' + str(i))

        for i in range(globalV.n_hl + 1):
            if (i == 0):
                self.w[str(i)] = tf.Variable(xavier([globalV.hl_size, 784]))
                self.b[str(i)] = tf.Variable(xavier([globalV.hl_size]))
                self.h[str(i)] = tf.matmul(self.X, tf.transpose(self.w[str(i)]))
                self.x[str(i)] = tf.nn.relu(tf.add(self.h[str(i)], self.b[str(i)]))
                if (globalV.update_rule == 'np-hybrid'):
                    self.s[str(i)] = tf.random_normal(shape=[globalV.batch_size, globalV.hl_size], mean=0, stddev=globalV.sigma,
                                                      dtype=tf.float32,
                                                      name='s_' + str(i), seed=globalV.seed)
                if (globalV.update_rule == 'ip'):
                    self.s['in'] = tf.random_normal(shape=[globalV.batch_size, 784], mean=0, stddev=globalV.sigma,
                                                    dtype=tf.float32,
                                                    name='s_in', seed=globalV.seed)

            elif (i < globalV.n_hl):
                self.w[str(i)] = tf.Variable(xavier([globalV.hl_size, globalV.hl_size]))
                self.b[str(i)] = tf.Variable(xavier([globalV.hl_size]))
                self.h[str(i)] = tf.matmul(self.x[str(i - 1)], tf.transpose(self.w[str(i)]))
                self.x[str(i)] = tf.nn.relu(tf.add(self.h[str(i)], self.b[str(i)]))
                if (globalV.update_rule == 'np-hybrid'):
                    self.s[str(i)] = tf.random_normal(shape=[globalV.batch_size, globalV.hl_size], mean=0, stddev=globalV.sigma,
                                                      dtype=tf.float32,
                                                      name='s_' + str(i), seed=globalV.seed)
            else:
                self.w[str(i)] = tf.Variable(xavier([10, globalV.hl_size]))
                self.b[str(i)] = tf.Variable(xavier([10]))
                self.h[str(i)] = tf.matmul(self.x[str(i - 1)], tf.transpose(self.w[str(i)]))
                self.pred = tf.nn.softmax(tf.add(self.h[str(i)], self.b[str(i)]))
                if (globalV.update_rule == 'np-hybrid'):
                    self.s[str(i)] = tf.random_normal(shape=[globalV.batch_size, 10], mean=0, stddev=globalV.sigma,
                                                      dtype=tf.float32,
                                                      name='s_' + str(i), seed=globalV.seed)


