import os
import struct
import torch
import numpy
import numpy as np
from numpy.core.fromnumeric import shape
from numpy import load
from torchvision import datasets, transforms

numpy.set_printoptions(threshold=np.inf)



#下载数据集
train_dataset = datasets.MNIST(root = 'data/', train = True, 
                               transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'data/', train = False, 
                               transform = transforms.ToTensor(), download = True)
##加载数据集
def load_minist(path, kind='train'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               %kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               %kind)
    with open(labels_path,'rb') as lbpath:
        magic,n =struct.unpack('>II',
                               lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb')as imgpath:
        magic,num, row, cols = struct.unpack('>IIII',
                                             imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


class_num = 2

[X_train, Y_train] = load_minist("data/MNIST/raw")
[X_test, Y_test] = load_minist("data/MNIST/raw", 't10k')
##################################################################
# print(X_train.shape,Y_train.shape)
# print(X_test.shape,Y_test.shape)
# (60000, 784) (60000,)       训练集60000张图片
# (10000, 784) (10000,)       测试集10000张图片
##################################################################
one_train = X_train[Y_train==1]      #图片为1的分为一类
two_train = X_train[Y_train==0]      #图片为0的分为一类
one_test = X_test[Y_test==1]         
two_test = X_test[Y_test==0]


# one_train = X_train[Y_train==1]      #图片为1的分为一类
# two_train = X_train[Y_train!=1]      #图片不为1的分为一类
# one_test = X_test[Y_test==1]         
# two_test = X_test[Y_test!=1]
##################################################################
# print(one_train.shape,two_train.shape)
# 输出(6742, 784) (5958, 784)
# 即 6742个 “1” 和 53258 个其他数字

##################################################################
#先验概率 = 类别出现次数 / 数据集总类多少
prior_1 = one_train.shape[0] / (one_train.shape[0] + two_train.shape[0])
prior_2 = two_train.shape[0] / (one_train.shape[0] + two_train.shape[0])
##################################################################
# print(prior_1,prior_2)
# 输出0.11236666666666667 0.8876333333333334
# 即 P(w_1)=0.11236666666666667
#    P(w_2)=0.8876333333333334

##################################################################

#u_1 = np.zeros(one_train.shape[0])
#u_2 = np.zeros(two_train.shape[0])
u_1 = np.expand_dims(np.mean(one_train, axis=0), axis=0)
u_2 = np.expand_dims(np.mean(two_train, axis=0), axis=0)
# mean压缩行，对各列求均值
# expand_dims扩展维度
##################################################################
# print(shape(u_1))
# # 输出 (1,784)
##################################################################
x_u_1 = one_train - numpy.repeat(u_1, one_train.shape[0], axis=0)
x_u_2 = two_train - numpy.repeat(u_2, two_train.shape[0], axis=0)

sigma_1 = np.dot(x_u_1.T, x_u_1) / one_train.shape[0] + np.eye(one_train.shape[1]) * 0.001
sigma_2 = np.dot(x_u_2.T, x_u_2) / two_train.shape[0] + np.eye(two_train.shape[1]) * 0.001

sigma = (np.dot(x_u_1.T, x_u_1) + np.dot(x_u_2.T, x_u_2)) / (one_train.shape[0] + two_train.shape[0]) + 0.001 * np.eye(one_train.shape[1])

inv_sigma_1=np.linalg.inv(sigma_1)
inv_sigma_2=np.linalg.inv(sigma_2)
inv_sigma  = np.linalg.inv(sigma)




##################################################################
#############################  LDF  ##############################
##################################################################
w_1 = np.dot(inv_sigma, u_1.T)
w_2 = np.dot(inv_sigma, u_2.T)

w_10 = -0.5 * np.dot(u_1, np.dot(inv_sigma, u_1.T)) + np.log(prior_1)
w_20 = -0.5 * np.dot(u_2, np.dot(inv_sigma, u_2.T)) + np.log(prior_2)

#print(two_test.shape[0])
g_1 = np.zeros((two_test.shape[0],2))
g_2 = np.zeros((two_test.shape[0],2))

for j in range(two_test.shape[1]):
    g_1[j,] = np.dot(w_1.T, two_test[j, :]) + w_10

for j in range(two_test.shape[1]):
    g_2[j] = np.dot(w_2.T, two_test[j, :]) + w_20
g_1 = np.expand_dims(g_1, axis=0)
g_2 = np.expand_dims(g_2, axis=0)
g = numpy.concatenate((g_1,g_2),axis=0)

ratio = sum((g==np.max(g,axis=0))[1]) / (g==np.max(g,axis=0))[1].shape


print('LDF准确率：',100*ratio[0],'%')

##################################################################
#############################  QDF  ##############################
##################################################################
# np.linalg.inv() 
W_1 = -0.5*inv_sigma_1
W_2 = -0.5*inv_sigma_2
# w_1 = np.dot(inv_sigma, u_1.T)
# w_2 = np.dot(inv_sigma, u_2.T)
w_1 = np.dot(inv_sigma_1,u_1.T)
w_2 = np.dot(inv_sigma_2,u_2.T)
# w_10 = -0.5 * np.dot(u_1, np.dot(inv_sigma, u_1.T)) + np.log(prior_1)
# w_20 = -0.5 * np.dot(u_2, np.dot(inv_sigma, u_2.T)) + np.log(prior_2)
w_10=-0.5*np.dot(u_1,np.dot(inv_sigma_1,u_1.T))-0.5*np.log(np.linalg.det(sigma_1))+np.log(prior_1)
w_20=-0.5*np.dot(u_2,np.dot(inv_sigma_2,u_2.T))-0.5*np.log(np.linalg.det(sigma_1))+np.log(prior_2)
#print(two_test.shape[0])
g_1 = np.zeros((two_test.shape[0],2))
g_2 = np.zeros((two_test.shape[0],2))

for j in range(two_test.shape[1]):
    #  g_1[j,] = np.dot(w_1.T, two_test[j, :]) + w_10
#    g_1[j,] = np.dot(np.dot(two_test[j, :].T,W_1), two_test[j, :]) +np.dot(w_1.T,two_test[j, :]) + w_10
    g_1[j,] = np.dot(np.dot(two_test[j, :].T,W_1), two_test[j, :]) +np.dot(w_1.T,two_test[j, :]) + w_10

for j in range(two_test.shape[1]):
    # g_2[j] = np.dot(w_2.T, two_test[j, :]) + w_20
    # g_2[j,] = np.dot(np.dot(two_test[j, :].T,W_2), two_test[j, :]) +np.dot(w_2.T,two_test[j, :]) + w_20
    g_2[j,] = np.dot(np.dot(two_test[j, :].T,W_2), two_test[j, :]) +np.dot(w_2.T,two_test[j, :]) + w_20

g_1 = np.expand_dims(g_1, axis=0)
g_2 = np.expand_dims(g_2, axis=0)
g = numpy.concatenate((g_1,g_2),axis=0)

ratio = sum((g==np.max(g,axis=0))[1]) / (g==np.max(g,axis=0))[1].shape


print('QDF准确率：',100*ratio[0],'%')