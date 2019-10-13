import matplotlib.pyplot as plt

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