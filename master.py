
# coding: utf-8

# In[34]:
import tensorflow as tf
import numpy as np
import time
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from makeNN import tfGraph
import globalV
from fileWriter import write_in_file
np.set_printoptions(precision=4, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# In[6]:


mnist = tf.keras.datasets.mnist
(x_train_data, y_train_cold),(x_test_data,y_test_cold) = mnist.load_data()
x_train_data, x_test_data = x_train_data / 255.0, x_test_data / 255.0


# In[7]:


with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train_cold,10))
    y_test =  sess.run(tf.one_hot(y_test_cold,10))


# In[8]:


x_train=[]
x_test=[]

n_train = len(x_train_data)
n_test = len(x_test_data)

for i in range(len(x_train_data)):
    x_train.append(np.ndarray.flatten(x_train_data[i]))
    
for i in range(len(x_test_data)):
    x_test.append(np.ndarray.flatten(x_test_data[i]))


# In[9]:


x_train = np.reshape(x_train,[n_train,784])
x_test = np.reshape(x_test,[n_test, 784])


def find_acc(which_acc):
    acc=[]
    if(which_acc=='test'):
        acc.append(sess.run('accuracy:0', feed_dict={'x:0':x_test, 'y:0':y_test}))

    elif(which_acc=='train'):
        n=int(n_train/n_test)
        for i in range(n):
            acc.append(sess.run('accuracy:0', feed_dict={'x:0':x_train[i*n_test:(i+1)*n_test], 'y:0':y_train[i*n_test:(i+1)*n_test]}))
    else:
        print('wrong accuracy requested!!')
    return np.mean(acc)

def print_acc():
    # train_acc.append(find_acc('train'))
    globalV.test_acc.append(find_acc('test'))
    print('epoch : ', i+1, '  test_acc : ', globalV.test_acc[-1])
    sec=int(time.time()-start)
    print(int(sec/60),'m ', int(sec%60),'s')

    return globalV.test_acc[-1]


# In[14]:
def train_network(num_hidden_layers):
    for j in range(int(n_train/globalV.batch_size)):
        ind = np.random.randint(0, n_train, size=(globalV.batch_size))

        if(globalV.update_rule=='sgd' or globalV.update_rule=='np-hybrid'):
            ops=['optimizer']
            sess.run(ops, feed_dict = {'x:0':x_train[ind], 'y:0':y_train[ind]})

        if(globalV.update_rule=='ip' or globalV.update_rule=='np' or globalV.update_rule=='np-hybrid'):
            updates=[]
            squiggles=[]
            for i in range(num_hidden_layers+1):
                updates.append('update_w'+str(i)+':0')
                updates.append('update_b'+str(i)+':0')
                squiggles.append('s_'+str(i)+':0')

            ops=[squiggles, updates]
            sess.run(ops, feed_dict = {'x:0':x_train[ind], 'y:0':y_train[ind]})

    return

# alphaZero is node perturbation.
failed=False

# In[17]:
start=time.time()
globalV.initialize()
dir='/nfs/nhome/live/yashm/Desktop/code/git/perturbations/np-hybrid.csv'

# globalV.learning_rate=5.0
globalV.n_seeds=1
globalV.alpha=1
globalV.learning_rate=5
globalV.n_hl=2
globalV.n_epochs=50

for jj in range(globalV.n_seeds):
    globalV.resetAcc()
    #globalV.seed=np.random.randint(0,1000)

    tf.set_random_seed(globalV.seed)
    np.random.seed(globalV.seed)
    objNN = tfGraph()
    objNN.build_graph()

    with tf.Session(graph=objNN.g) as sess:
        sess.run(tf.global_variables_initializer())

        print('TRAINING', '\nwidth : ', globalV.hl_size, '\ndepth : ', globalV.n_hl, '\n')
        for i in range(globalV.n_epochs):
            if i%globalV.interval==0:
                if(print_acc()>98):
                    print('98% achieved in - ',i,' epochs')
                    break
                if (i>50 and find_acc('test')<20) or (i>500 and find_acc('test')<35):
                    failed=True
                    break
            train_network(globalV.n_hl)

        sess.close()
        if(globalV.writeFile):
            write_in_file(failed, dir)

        failed=False

    tf.reset_default_graph()
        
sec=int(time.time()-start)
print('TOTAL TIME : ', int(sec/60),'m ', int(sec%60),'s')


