import tensorflow as tf
import globalV
from attrdict import AttrDict

def my_network(X):
    w = {}
    b = {}
    h = {}
    x = {}
    s = {}
    h_star = {}
    x_star = {}

    xavier = tf.compat.v1.initializers.glorot_normal(seed=globalV.seed)

    for i in range(globalV.n_hl + 1):
        s[str(i)] = tf.zeros([1], dtype=tf.float32, name='s_' + str(i))

    for i in range(globalV.n_hl + 1):
        if (i == 0):
            w[str(i)] = tf.Variable(xavier([globalV.hl_size, 784]))
            b[str(i)] = tf.Variable(xavier([globalV.hl_size]))
            h[str(i)] = tf.matmul(X, tf.transpose(w[str(i)]))
            x[str(i)] = tf.nn.relu(tf.add(h[str(i)], b[str(i)]))
            s[str(i)] = tf.random.normal(shape=[globalV.batch_size, globalV.hl_size], mean=0, stddev=globalV.sigma,
                                                dtype=tf.float32,
                                                name='s_' + str(i), seed=globalV.seed)
            

        elif (i < globalV.n_hl):
            w[str(i)] = tf.Variable(xavier([globalV.hl_size, globalV.hl_size]))
            b[str(i)] = tf.Variable(xavier([globalV.hl_size]))
            h[str(i)] = tf.matmul(x[str(i - 1)], tf.transpose(w[str(i)]))
            x[str(i)] = tf.nn.relu(tf.add(h[str(i)], b[str(i)]))
            s[str(i)] = tf.random.normal(shape=[globalV.batch_size, globalV.hl_size], mean=0, stddev=globalV.sigma,
                                                dtype=tf.float32,
                                                name='s_' + str(i), seed=globalV.seed)

        else:
            w[str(i)] = tf.Variable(xavier([10, globalV.hl_size]))
            b[str(i)] = tf.Variable(xavier([10]))
            h[str(i)] = tf.matmul(x[str(i - 1)], tf.transpose(w[str(i)]))
            pred = tf.nn.softmax(tf.add(h[str(i)], b[str(i)]), name='pred')
            
            s[str(i)] = tf.random.normal(shape=[globalV.batch_size, 10], mean=0, stddev=globalV.sigma,
                                                dtype=tf.float32,
                                                name='s_' + str(i), seed=globalV.seed)

    for i in range(globalV.n_hl + 1):
        if (i == 0):
            h_star[str(i)] = tf.matmul(X, tf.transpose(w[str(i)]))
        else:
            h_star[str(i)] = tf.matmul(x_star[str(i - 1)], tf.transpose(w[str(i)]))

        if (i < globalV.n_hl):
            x_star[str(i)] = tf.nn.relu(tf.add(tf.add(h_star[str(i)], b[str(i)]), s[str(i)]))

        else:
            pred_star = tf.nn.softmax(tf.add(tf.add(h_star[str(i)], b[str(i)]), s[str(i)]))
                
    return pred, pred_star, (w, s, b, x)
    


def define_graph():
    tf.compat.v1.reset_default_graph()

    X=tf.compat.v1.placeholder(shape=[None, 784], dtype=tf.float32)
    Y=tf.compat.v1.placeholder(shape=[None, 10], dtype=tf.float32)
    
    pred, pred_star, network_params=my_network(X)
    (w, s, b, x)=network_params
    update_w = {}
    update_b = {}
    del_h = {}
    w_norm={}
    b_sum={}
    delta_w = {}
    delta_b = {}
    error = tf.reduce_mean(tf.math.square(pred - Y), 1)
    mse =  tf.reduce_mean(error)
    error_star = tf.reduce_mean(tf.math.square(pred_star - Y), 1)
    var = globalV.sigma ** 2

    k = - globalV.lr*(error_star - error) / var

    for i in range(globalV.n_hl + 1):
    
        del_h[str(i)] = s[str(i)]

        if (i == 0):
            delta_w[str(i)] = tf.einsum('ki,kj->kij', del_h[str(i)], X)

        else:
            delta_w[str(i)] = tf.einsum('ki,kj->kij', del_h[str(i)], x[str(i - 1)])

        delta_w[str(i)] = tf.einsum('kij,k->kij', delta_w[str(i)], k)
        delta_w[str(i)] = tf.reduce_mean(delta_w[str(i)], 0)
        delta_b[str(i)] = tf.einsum('ki,k->ki', del_h[str(i)], k)
        delta_b[str(i)] = tf.reduce_mean(delta_b[str(i)], 0)

        update_w[str(i)] = tf.compat.v1.assign(w[str(i)], tf.add(w[str(i)], delta_w[str(i)]), name='update_w' + str(i))
        update_b[str(i)] = tf.compat.v1.assign(b[str(i)], tf.add(b[str(i)], delta_b[str(i)]), name='update_b' + str(i))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=globalV.lr).minimize(mse)

    is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = 100 * tf.reduce_mean(tf.cast(is_correct, tf.float32))

    for i in range(globalV.n_hl + 1):
        w_norm[str(i)]=tf.norm(w[str(i)])
        b_sum[str(i)]=tf.math.reduce_sum(b[str(i)])
        
    return AttrDict(locals())