import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
import argparse
import csv
from makeNN import define_graph
from data_loader import load_mnist
import globalV
from utils import file_writer, sma_accuracy, parse_args, get_elapsed_time

np.set_printoptions(precision=4, suppress=True)

(x_train, y_train),(x_test, y_test)=load_mnist()

start=time.time()
dir_path='/nfs/ghome/live/yashm/Desktop/research/perturbations/results/learning_curves.csv'

tf.compat.v1.set_random_seed(globalV.seed)
np.random.seed(globalV.seed)

parse_args()

for hl in range(globalV.n_hl+1):
    globalV.w_norm[str(hl)]=[]
    globalV.b_sum[str(hl)]=[]

graph=define_graph()
print('built with CUDA: ', tf.test.is_built_with_cuda())
print('is using GPU: ', tf.test.is_gpu_available())

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('TRAINING\ntrainable variables : ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    
    for i in range(globalV.n_epochs):
        if(i%globalV.interval==0):
            globalV.test_acc.append(sess.run(graph.accuracy, feed_dict = {graph.X:x_test, graph.Y:y_test}))
            print(globalV.test_acc[-1])
            for hl in range(globalV.n_hl+1):
                globalV.w_norm[str(hl)].append(sess.run(graph.w_norm[str(hl)]))
                globalV.b_sum[str(hl)].append(sess.run(graph.b_sum[str(hl)]))


        for j in range(int(globalV.n_train/globalV.batch_size)):
            ind = np.random.randint(0, globalV.n_train, size=(globalV.batch_size))    
            if(globalV.update_rule=='sgd'):
                ops=graph.optimizer
            elif(globalV.update_rule=='np'):
                ops=[globalV.squiggles, globalV.updates]

            # if(j%10==0):
            #     globalV.test_acc.append(sess.run(graph.accuracy, feed_dict = {graph.X:x_test, graph.Y:y_test}))
            #     print(globalV.test_acc[-1])
            #     for hl in range(globalV.n_hl+1):
            #         globalV.w_norm[str(hl)].append(sess.run(graph.w_norm[str(hl)]))
            #         globalV.b_sum[str(hl)].append(sess.run(graph.b_sum[str(hl)]))
            

            sess.run(ops, feed_dict = {graph.X:x_train[ind], graph.Y:y_train[ind]})
        
sess.close()   

elapsed_time=get_elapsed_time(time.time()-start)
print('SEED: ', globalV.seed)
print(elapsed_time)

if(globalV.writeFile):
    file_writer(dir_path, elapsed_time, write_norms=True)



# print('\nwidth : ', globalV.hl_size, '\ndepth : ', globalV.n_hl, '\nlearning rate : ', globalV.lr)
# print('\nwidth : ', globalV.hl_size, '\ndepth : ', globalV.n_hl, '\nlearning rate : ', globalV.lr)



