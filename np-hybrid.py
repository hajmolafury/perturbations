
# coding: utf-8

# In[34]:
import tensorflow as tf
import numpy as np
import time
import csv
from makeNN import tfGraph
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=4, suppress=True)


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


# In[10]:


sigma=0.001
batch_size=1000


# In[12]:


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


# In[13]:


def print_acc():    
    # train_acc.append(find_acc('train'))
    test_acc.append(find_acc('test'))
    print('epoch : ', i+1, '  test_acc : ', test_acc[-1])
    sec=int(time.time()-start)
    print(int(sec/60),'m ', int(sec%60),'s')
    
    return test_acc[-1] 


# In[14]:


def train_network(num_hidden_layers):
    for j in range(int(n_train/batch_size)):
        ind = np.random.randint(0, n_train, size=(batch_size))
        
        if(update_rule=='sgd' or update_rule=='np-hybrid'):
            ops=['optimizer']
            sess.run(ops, feed_dict = {'x:0':x_train[ind], 'y:0':y_train[ind]})
            
        if(update_rule=='ip' or update_rule=='np' or update_rule=='np-hybrid'):
            updates=[]
            squiggles=[]
            for i in range(num_hidden_layers+1):
                updates.append('update_w'+str(i)+':0')
                updates.append('update_b'+str(i)+':0')
                squiggles.append('s_'+str(i)+':0')
            
            ops=[squiggles, updates]
            sess.run(ops, feed_dict = {'x:0':x_train[ind], 'y:0':y_train[ind]})

    return


# In[15]:


def write_in_file(failed):
    row=test_acc
    # print(test_acc)
    params=["depth", str(num_hidden_layers), "width", str(hl_size),"learning rate", str(learning_rate),"n_epochs", str(n_epochs), "alpha", str(alpha)]

    with open('Desktop/code/np-hybrid/hybrid-test.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        if(failed):
            row=["Failed"]
        writer.writerow(row)
        csvFile.flush()
        
        
    csvFile.close()
    return


# In[16]:


import sys

sys_input=False

if(sys_input):
    hl_size = int(sys.argv[2])
    # learning_rate = int(sys.argv[3])
    # num_hidden_layers= int(sys.argv[1])
    # n_seeds=int(sys.argv[5])
    # n_epochs=int(sys.argv[4])
    # alpha=float(sys.argv[6])
    # interval=5
    
else:
    n_epochs=3
    hl_size = 300
    learning_rate = 5.0
    num_hidden_layers= 1
    n_seeds=1
    alpha=1
    interval=1
    
update_rule='np-hybrid'
# alphaZero is node perturbation.
failed=False


# In[17]:


start=time.time()

for jj in range(n_seeds):
    test_acc=[]
    seed=np.random.randint(0,1000)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    objNN = tfGraph(alpha=1.0, learning_rate=0.5)
    objNN.build_graph()

    with tf.Session(graph=objNN.g) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run('change_lr:0')

        print('TRAINING : ', hl_size,' hidden units', '\nlearning rate : ', sess.run('lr:0'), '\n')
        for i in range(n_epochs):
            if i%interval==0:
                if(print_acc()>98):
                    print('98% achieved in - ',i,' epochs')
                    break
                if (i>50 and find_acc('test')<20) or (i>500 and find_acc('test')<35):
                    failed=True
                    break
            train_network(num_hidden_layers)

        sess.close()
        # write_in_file(failed)
        failed=False
        
sec=int(time.time()-start)
print('TOTAL TIME : ', int(sec/60),'m ', int(sec%60),'s')


# In[18]:


# fig, ax1 = plt.subplots(figsize=(8, 5.5))
# xx=np.arange(0,n_epochs, interval)
# ax1.set_xlabel('epochs')
# ax1.set_ylabel('accuracy')
# ax1.plot(xx, test_acc, color='tab:red', label='test accuracy')
# ax1.plot(xx, train_acc, color='tab:blue', label='training accuracy')
# ax1.legend()
# 
# axes = plt.gca()
# 
# ax1.set_facecolor("#ffffb3")
# 
# ax1.annotate(str(round(test_acc[-1],1)),xy=(xx[-1]-0.5,test_acc[-1]+0.7), color='tab:red')
# ax1.annotate(str(round(train_acc[-1],1)),xy=(xx[-1]-0.5,train_acc[-1]-2.2), color='tab:blue')
# 
# plt.grid(True, color="#93a1a1", alpha=0.3)
# plt.show()
# plt.savefig('plot1.svg', dpi=500)

