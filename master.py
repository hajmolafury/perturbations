
# coding: utf-8

# In[34]:
import tensorflow as tf
import numpy as np
import time
import argparse
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from makeNN import tfGraph
import globalV
from fileWriter import write_in_file
np.set_printoptions(precision=4, suppress=True)
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# In[6]:


mnist = tf.keras.datasets.mnist
(x_train_data, y_train_cold),(x_test_data,y_test_cold) = mnist.load_data()
x_train_data, x_test_data = x_train_data / 255.0, x_test_data / 255.0


# In[7]:

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

def sma_accuracy():
    return np.ma.average(globalV.test_acc[-10:])

def print_accuracy():
    print('epoch : ', i+1, '  test_acc : ', globalV.test_acc[-1])
    sec=int(time.time()-start)
    print(int(sec/60),'m ', int(sec%60),'s')
    return

# In[14]:
def train_network(num_hidden_layers):
    for j in range(int(n_train/globalV.batch_size)):
        ind = np.random.randint(0, n_train, size=(globalV.batch_size))

        ops=['optimizer']
        sess.run(ops, feed_dict = {'x:0':x_train[ind], 'y:0':y_train[ind]})

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

start=time.time()
globalV.initialize()
dir='/nfs/nhome/live/yashm/Desktop/code/git/perturbations/np-hybrid.csv'

ap = argparse.ArgumentParser()
ap.add_argument("--learning_rate")
ap.add_argument("--n_hl")
ap.add_argument("--hl_size")
ap.add_argument("--alpha")
args = vars(ap.parse_args())

globalV.learning_rate=float(args["learning_rate"])
globalV.n_hl=int(args["n_hl"])
globalV.hl_size=int(args["hl_size"])
globalV.alpha=float(args["alpha"])

globalV.n_seeds=5
globalV.n_epochs=3000
globalV.writeFile=True
onlyEpochs=False
failed = True
if(globalV.n_hl==1 and globalV.alpha<0.1):
    globalV.batch_size=250


for jj in range(globalV.n_seeds):
    failed = False
    globalV.resetAcc()
    firstCross=True
    globalV.seed=np.random.randint(0,1000)

    tf.set_random_seed(globalV.seed)
    np.random.seed(globalV.seed)
    objNN = tfGraph()
    objNN.build_graph()

    with tf.Session(graph=objNN.g) as sess:
        sess.run(tf.global_variables_initializer())
        print('TRAINING\nnumber of variables : ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        print('\nwidth : ', globalV.hl_size, '\ndepth : ', globalV.n_hl, '\nalpha : ', globalV.alpha, '\nlearning rate : ', globalV.learning_rate)
        for i in range(globalV.n_epochs):
            if (i > 250 and globalV.test_acc[-1] < 90):
                break

            if (i%globalV.interval==0):
                if(sma_accuracy()>97.9):
                    print('97.9% achieved in - ',i,' epochs')
                    globalV.epochs979.append(i)
                    failed=False
                    break

                elif (sma_accuracy() > 97.7 and firstCross):
                    print('97.7% achieved in - ', i, ' epochs')
                    globalV.epochs977.append(i)
                    firstCross=False

            globalV.test_acc.append(find_acc('test'))
            print_accuracy()
            train_network(globalV.n_hl)

        sess.close()
    tf.reset_default_graph()

if(globalV.writeFile):
    write_in_file(failed, dir, onlyEpochs)

print(globalV.epochs977)
print(globalV.epochs979)


sec=int(time.time()-start)
print('TOTAL TIME : ', int(sec/60),'m ', int(sec%60),'s')


