# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import trange
import numpy as np
import time
import argparse
import csv

from makeNN import tfGraph
import globalV
from fileWriter import write_in_file
np.set_printoptions(precision=4, suppress=True)


# %%
globalV.initialize()

# %%
def train_network(iterator):
    batches=100
    #print(sess.graph.get_operations())
    sess.run(iterator.initializer)

    updates=[]
    squiggles=[]
    for i in range(globalV.n_hl+1):
        updates.append('update_w'+str(i)+':0')
        updates.append('update_b'+str(i)+':0')
        squiggles.append('s_'+str(i)+':0')
    
    ops_sgd=['optimizer', 'mse']
    ops_np=[squiggles, updates]

    # t=trange(batches)
    for j in range(batches):
        _,loss_i=sess.run(ops_sgd)
        sess.run(ops_np)
        # t.set_postfix(loss='{!s:05.3f}'.format(loss_i))
    return


# %%
# alphaZero is node perturbation.

start=time.time()
globalV.initialize()

dir='/nfs/nhome/live/yashm/Desktop/code/git/perturbations/data/trial.csv'

globalV.n_seeds=1
globalV.n_epochs=5
globalV.writeFile=False
onlyEpochs=False
failed = True


# %%
for jj in range(globalV.n_seeds):
    globalV.resetAcc()
    globalV.seed=np.random.randint(0,1000)

    tf.compat.v1.set_random_seed(globalV.seed)
    np.random.seed(globalV.seed)
    objNN = tfGraph()
    iterator=objNN.build_graph()

    with tf.compat.v1.Session(graph=objNN.g) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print('TRAINING\nnumber of variables : ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
        print('\nwidth : ', globalV.hl_size, '\ndepth : ', globalV.n_hl, '\nalpha : ', globalV.alpha, '\nlearning rate : ', globalV.learning_rate)
        for i in range(globalV.n_epochs):
            train_network(iterator)
 

        sess.close()
    tf.compat.v1.reset_default_graph()

if(globalV.writeFile):
    write_in_file(failed, dir, onlyEpochs)

sec=int(time.time()-start)
print('TOTAL TIME : ', int(sec/60),'m ', int(sec%60),'s')


# %%


