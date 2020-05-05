import sys
import numpy as np

update_rule='sgd'
batch_size=100
n_epochs=5
n_hl=2
hl_size=300
lr=0.05
interval=1

# seed=np.random.randint(0,1000)
writeFile=False
seed=302
sigma=0.001
n_train=60000
n_test=10000


test_acc=[]
train_acc=[]
w_norm={}
b_sum={}

updates=[]
squiggles=[]
for jj in range(n_hl+1):
    updates.append('update_w'+str(jj)+':0')
    updates.append('update_b'+str(jj)+':0')
    squiggles.append('s_'+str(jj)+':0')