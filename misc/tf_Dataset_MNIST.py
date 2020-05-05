# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tqdm import trange


# %%
mnist = tfds.image.MNIST()

mnist.download_and_prepare()

datasets = mnist.as_dataset()
train_dataset, test_dataset = datasets['train'], datasets['test']

batch_size=100

train_dataset = train_dataset.batch(batch_size)
train_dataset=train_dataset.prefetch(1)
# Prefetch data for faster consumption:
#.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(1)

iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)

#* Iterator returns nested Tensors in the form of a dictionary with keys as 'image' and 'label' of size batch_size

batch_data = iterator.get_next()
images=tf.keras.layers.Flatten()(tf.dtypes.cast(batch_data['image'], tf.float32)/255)
labels=tf.one_hot(batch_data['label'],10)


# %%
nodes_hl1=300
nodes_hl2=300
n_classes=10

  
initializer= tf.compat.v1.initializers.glorot_normal()

hl1={'weights':tf.Variable(initializer([784,nodes_hl1])),'bias':tf.Variable(initializer([nodes_hl1]))}
hl2={'weights':tf.Variable(initializer([nodes_hl1,nodes_hl2])),'bias':tf.Variable(initializer([nodes_hl2]))}
output_layer={'weights':tf.Variable(initializer([nodes_hl2,n_classes])),'bias':tf.Variable(initializer([n_classes]))}


l1=tf.add(tf.matmul(images,hl1['weights']),hl1['bias'])
l1=tf.nn.relu(l1)
                                                                            
l2=tf.add(tf.matmul(l1,hl2['weights']),hl2['bias'])
l2=tf.nn.relu(l2)
                                                                                
logits=tf.add(tf.matmul(l2,output_layer['weights']),output_layer['bias'])
                                                                                


# %%
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)
# get accuracy
pred = tf.argmax(logits, 1)
equality = tf.equal(pred, tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
init_op = tf.global_variables_initializer()
# run the training
epochs=2
n_train=60000
batches = int(n_train/batch_size)

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(epochs):
        sess.run(iterator.initializer)
        t=trange(batches)
        for j in t:
            _,loss_i=sess.run([optimizer, loss])
            print(loss_i)
            # t.set_postfix(loss='{:05.3f}'.format(loss_i))


# %%


