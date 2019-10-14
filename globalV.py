import sys

def initialize(sys_input=False):
    # if (sys_input):
    #     global hl_size = int(sys.argv[2])
    #     global learning_rate = int(sys.argv[3])
    #     global num_hidden_layers= int(sys.argv[1])
    #     global n_seeds=int(sys.argv[5])
    #     global n_epochs=int(sys.argv[4])
    #     global alpha=float(sys.argv[6])
    #     global interval=5

    # else:
    global n_epochs
    global test_acc
    global train_acc
    global n_seeds
    global interval
    global seed
    global sigma
    global batch_size
    global hl_size
    global learning_rate
    global n_hl
    global update_rule
    global alpha
    global writeFile
    global epochs977
    global epochs979

    sigma=0.001
    batch_size=1000
    update_rule='np-hybrid'
    test_acc=[]
    train_acc=[]
    epochs977 = []
    epochs979=[]

    seed=0
    alpha=1
    writeFile=False

    if(sys_input):
        n_epochs = int(sys.argv[4])
        n_seeds = int(sys.argv[5])
        n_hl=int(sys.argv[1])
        hl_size=int(sys.argv[2])
        learning_rate=float(sys.argv[3]) #have to change this to float!
        alpha=float(sys.argv[6])
        interval = 3

    else:
        n_epochs=3
        n_seeds=1
        n_hl=1
        hl_size=300
        learning_rate=1
        alpha=1.0
        interval=1

def resetAcc():
    test_acc.clear()
    train_acc.clear()