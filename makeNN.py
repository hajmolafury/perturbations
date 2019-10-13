import tensorflow as tf
import numpy as np


class tfGraph:

    def __init__(self, alpha, learning_rate, update_rule='np-hybrid', n_hl=300, num_hidden_layers=1, seed=0, batch_size=1000):

        self.sigma = 0.001
        self.batch_size=batch_size
        self.alpha = alpha
        self.num_hidden_layers=num_hidden_layers
        self.update_rule=update_rule
        self.n_hl=n_hl
        self.seed=seed
        self.learning_rate=learning_rate
        self.g = tf.Graph()
        # self.build_graph()


    def build_graph(self):

        with self.g.as_default():

            self.define_variables()
            error = tf.reduce_mean(np.square(self.pred - self.Y), 1)
            mse = tf.losses.mean_squared_error(predictions=self.pred, labels=self.Y)

            if (self.update_rule == 'ip' or self.update_rule == 'np' or self.update_rule == 'np-hybrid'):
                for i in range(self.num_hidden_layers + 1):
                    if (i == 0):
                        self.h_star[str(i)] = tf.matmul(tf.add(self.X, self.s['in']), tf.transpose(self.w[str(i)]))
                    else:
                        self.h_star[str(i)] = tf.matmul(self.x_star[str(i - 1)], tf.transpose(self.w[str(i)]))

                    if (i < self.num_hidden_layers):
                        self.x_star[str(i)] = tf.nn.relu(tf.add(tf.add(self.h_star[str(i)], self.b[str(i)]), self.s[str(i)]))

                    else:
                        pred_star = tf.nn.softmax(tf.add(tf.add(self.h_star[str(i)], self.b[str(i)]), self.s[str(i)]))

                error_star = tf.reduce_mean(np.square(pred_star - self.Y), 1)
                var = self.sigma ** 2
                k = -self.learning_rate * (error_star - error) / var

                for i in range(self.num_hidden_layers + 1):
                    if (self.update_rule == 'ip'):
                        self.del_h[str(i)] = self.h_star[str(i)] - self.h[str(i)]

                    elif (self.update_rule == 'np'):
                        self.del_h[str(i)] = self.s[str(i)]

                    elif (self.update_rule == 'np-hybrid'):
                        self.del_h[str(i)] = self.s[str(i)]
                        k = k * (1 - self.alpha)

                    if (i == 0):
                        self.delta_w[str(i)] = tf.einsum('ki,kj->kij', self.del_h[str(i)], self.X)

                    else:
                        self.delta_w[str(i)] = tf.einsum('ki,kj->kij', self.del_h[str(i)], self.x[str(i - 1)])

                    self.delta_w[str(i)] = tf.einsum('kij,k->kij', self.delta_w[str(i)], k)
                    self.delta_w[str(i)] = tf.reduce_mean(self.delta_w[str(i)], 0)
                    self.delta_b[str(i)] = tf.einsum('ki,k->ki', self.del_h[str(i)], k)
                    self.delta_b[str(i)] = tf.reduce_mean(self.delta_b[str(i)], 0)

                    self.update_w[str(i)] = tf.assign(self.w[str(i)], tf.add(self.w[str(i)], self.delta_w[str(i)]), name='update_w' + str(i))
                    self.update_b[str(i)] = tf.assign(self.b[str(i)], tf.add(self.b[str(i)], self.delta_b[str(i)]), name='update_b' + str(i))

            self.change_lr = tf.assign(self.lr, tf.reshape(self.learning_rate, [1]), name='change_lr')

            if (self.update_rule == 'np-hybrid'):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha * self.lr[0], name='optimizer').minimize(mse)
            elif (self.update_rule == 'sgd'):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr[0], name='optimizer').minimize(mse)
            # train_step=optimizer.minimize(mse, name='train_step')

            is_correct = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
            accuracy = 100 * tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.identity(accuracy, 'accuracy')

        # return self.g


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

        xavier = tf.contrib.layers.xavier_initializer(seed=self.seed)
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')
        self.lr = tf.Variable(xavier([1]), dtype=tf.float32, name='lr')

        self.s['in'] = tf.zeros([1], dtype=tf.float32, name='s_in')

        for i in range(self.num_hidden_layers + 1):
            self.s[str(i)] = tf.zeros([1], dtype=tf.float32, name='s_' + str(i))

        for i in range(self.num_hidden_layers + 1):
            if (i == 0):
                self.w[str(i)] = tf.Variable(xavier([self.n_hl, 784]))
                self.b[str(i)] = tf.Variable(xavier([self.n_hl]))
                self.h[str(i)] = tf.matmul(self.X, tf.transpose(self.w[str(i)]))
                self.x[str(i)] = tf.nn.relu(tf.add(self.h[str(i)], self.b[str(i)]))
                if (self.update_rule == 'np' or self.update_rule == 'np-hybrid'):
                    self.s[str(i)] = tf.random_normal(shape=[self.batch_size, self.n_hl], mean=0, stddev=self.sigma,
                                                      dtype=tf.float32,
                                                      name='s_' + str(i), seed=self.seed)
                if (self.update_rule == 'ip'):
                    self.s['in'] = tf.random_normal(shape=[self.batch_size, 784], mean=0, stddev=self.sigma,
                                                    dtype=tf.float32,
                                                    name='s_in', seed=self.seed)

            elif (i < self.num_hidden_layers):
                self.w[str(i)] = tf.Variable(xavier([self.n_hl, self.n_hl]))
                self.b[str(i)] = tf.Variable(xavier([self.n_hl]))
                self.h[str(i)] = tf.matmul(self.x[str(i - 1)], tf.transpose(self.w[str(i)]))
                self.x[str(i)] = tf.nn.relu(tf.add(self.h[str(i)], self.b[str(i)]))
                if (self.update_rule == 'np' or self.update_rule == 'np-hybrid'):
                    self.s[str(i)] = tf.random_normal(shape=[self.batch_size, self.n_hl], mean=0, stddev=self.sigma,
                                                      dtype=tf.float32,
                                                      name='s_' + str(i), seed=self.seed)
            else:
                self.w[str(i)] = tf.Variable(xavier([10, self.n_hl]))
                self.b[str(i)] = tf.Variable(xavier([10]))
                self.h[str(i)] = tf.matmul(self.x[str(i - 1)], tf.transpose(self.w[str(i)]))
                self.pred = tf.nn.softmax(tf.add(self.h[str(i)], self.b[str(i)]))
                if (self.update_rule == 'np' or self.update_rule == 'np-hybrid'):
                    self.s[str(i)] = tf.random_normal(shape=[self.batch_size, 10], mean=0, stddev=self.sigma,
                                                      dtype=tf.float32,
                                                      name='s_' + str(i), seed=self.seed)


