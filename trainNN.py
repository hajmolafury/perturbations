# print(args.values())
# def find_acc(sess, which_acc):
#     acc=[]
#     if(which_acc=='test'):
#         acc.append(sess.run('accuracy:0', feed_dict={'x:0':x_test, 'y:0':y_test}))
#
#     elif(which_acc=='train'):
#         n=int(n_train/n_test)
#         for i in range(n):
#             acc.append(sess.run('accuracy:0', feed_dict={'x:0':x_train[i*n_test:(i+1)*n_test], 'y:0':y_train[i*n_test:(i+1)*n_test]}))
#     else:
#         print('wrong accuracy requested!!')
#     return np.mean(acc)