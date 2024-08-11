#from keras.models import Model, load_model, Sequential
#from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, Concatenate
#from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import pathlib
import gc
from datetime import timedelta
import os
import scipy.io as sio
#from keras.callbacks import ModelCheckpoint
#from tcn import TCN
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
#from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import LSTM
import tensorflow.keras.backend as K
import pdb
import tensorflow as tf
import time

# 清除之前的计算图
tf.keras.backend.clear_session()

#train = '/content/drive/MyDrive/Colab Notebooks/2023-5-11-multihead-test/train_AUV_multihead.mat'
train = 'train_AUV_multihead.mat'
train_dataset = sio.loadmat(train)
train_data_v = train_dataset['train_data_v']
train_data_hpr = train_dataset['train_data_hpr']
train_data_acc = train_dataset['train_data_acc']
train_data_w = train_dataset['train_data_w']

#test = '/content/drive/MyDrive/Colab Notebooks/2023-5-11-multihead-test/test_AUV_multihead_2019_4_28_13_19_07.mat'
test = 'test_AUV_multihead_2019_10_21_17_8_08.mat'
test_dataset = sio.loadmat(test)
test_data_v = test_dataset['test_data_v']
test_data_hpr = test_dataset['test_data_hpr']
test_data_acc = test_dataset['test_data_acc']
test_data_w = test_dataset['test_data_w']
print(test_data_v.shape)

test_label = test_dataset['test_label'].reshape([-1, 2])
test_data_len = test_label.shape[0]
print(test_data_len)

X1_train_v = train_data_v.reshape(-1,4,3)
X1_train_hpr = train_data_hpr.reshape(-1,20,3)
X1_train_acc = train_data_acc.reshape(-1,20,3)
X1_train_w = train_data_w.reshape(-1,20,3)
X1_test_v = test_data_v.reshape(-1,4,3)
X1_test_hpr = test_data_hpr.reshape(-1,20,3)
X1_test_acc = test_data_acc.reshape(-1,20,3)
X1_test_w = test_data_w.reshape(-1,20,3)
Y1_test = test_label.reshape(-1,2)

mean = np.mean(X1_train_v, axis=0)  # 均值
X1_test_v -= mean  # 测试集
std = np.std(X1_train_v, axis=0) # 标准差
X1_test_v /= std  # 测试集

# 计算最小值和最大值
# min_val = np.min(X1_train_v, axis=0)
# max_val = np.max(X1_train_v, axis=0)
# # 最大最小值归一化
# X1_test_v = (X1_test_v - min_val) / (max_val - min_val)

mean = np.mean(X1_train_hpr, axis=0)  # 均值
X1_test_hpr -= mean  # 测试集
std = np.std(X1_train_hpr, axis=0) # 标准差
X1_test_hpr /= std  # 测试集

# # 计算最小值和最大值
# min_val = np.min(X1_train_hpr, axis=0)
# max_val = np.max(X1_train_hpr, axis=0)
# # 最大最小值归一化
# X1_test_hpr = (X1_test_hpr - min_val) / (max_val - min_val)

mean = np.mean(X1_train_acc, axis=0)  # 均值
X1_test_acc -= mean  # 测试集
std = np.std(X1_train_acc, axis=0) # 标准差
X1_test_acc /= std  # 测试集

# 计算最小值和最大值
# min_val = np.min(X1_train_acc, axis=0)
# max_val = np.max(X1_train_acc, axis=0)
# # 最大最小值归一化
# X1_test_acc = (X1_test_acc - min_val) / (max_val - min_val)

mean = np.mean(X1_train_w, axis=0)  # 均值
X1_test_w -= mean  # 测试集
std = np.std(X1_train_w, axis=0) # 标准差
X1_test_w /= std  # 测试集

# # 计算最小值和最大值
# min_val = np.min(X1_train_w, axis=0)
# max_val = np.max(X1_train_w, axis=0)
# # 最大最小值归一化
# X1_test_w = (X1_test_w - min_val) / (max_val - min_val)

epoch = 200
model_name = 'model_' + str(epoch) + '.hdf5'
#model=load_model(model_name)
input_v = Input(shape=(4,3))
input_hpr = Input(shape=(20,3))
input_acc = Input(shape=(20,3))
input_w = Input(shape=(20,3))

def build_model():
    x = Conv1D(filters=32,
                kernel_size=3,
                padding='same',
                activation='relu')(input_v)#input_shape=(timesteps, features)
    x = Conv1D(filters=64,
                kernel_size=3,
                padding='same',
                activation='relu')(x)
    x = Conv1D(filters=128,
                kernel_size=3,
                padding='same',
                activation='relu')(x)
    x = Conv1D(filters=256,
                kernel_size=3,
                padding='same',
                activation='relu')(x)
    x = BatchNormalization()(x)
    residual1 = Conv1D(filters=256, kernel_size=1, padding='same')(input_v) # 输入数据的维度需要与输出的维度相同
    x = Add()([x, residual1])
    x = UpSampling1D(size=5)(x)
    # x = Bidirectional(LSTM(256, return_sequences=True))(x)
    # x = Bidirectional(LSTM(256, return_sequences=True))(x)
    # x = LSTM(128, return_sequences=True)(x)
    # x = LSTM(128, return_sequences=True)(x)
    #x = Model(inputs = input_v, outputs = x)

    #for dilation_rate in dilation_rates:
    y = Conv1D(filters=32,
                kernel_size=3,
                padding='same',
                activation='relu')(input_hpr)
    y = Conv1D(filters=64,
                kernel_size=3,
                padding='same',
                activation='relu')(y)
    y = Conv1D(filters=128,
                kernel_size=3,
                padding='same',
                activation='relu')(y)
    y = Conv1D(filters=256,
                kernel_size=3,
                padding='same',
                activation='relu')(y)
    y = BatchNormalization()(y)
    residual2 = Conv1D(filters=256, kernel_size=1, padding='same')(input_hpr) # 输入数据的维度需要与输出的维度相同
    y = Add()([y, residual2])
    # y = Bidirectional(LSTM(256, return_sequences=True))(y)
    # y = Bidirectional(LSTM(256, return_sequences=True))(y)
    # y = LSTM(64, return_sequences=True)(y)
    # y = LSTM(64, return_sequences=True)(y)
    #y = Model(inputs=input_hpr, outputs=y)

    #for dilation_rate in dilation_rates:
    z = Conv1D(filters=32,
                kernel_size=3,
                padding='same',
                activation='relu')(input_acc)
    z = Conv1D(filters=64,
                kernel_size=3,
                padding='same',
                activation='relu')(z)
    z = Conv1D(filters=128,
                kernel_size=3,
                padding='same',
                activation='relu')(z)
    z = Conv1D(filters=256,
                kernel_size=3,
                padding='same',
                activation='relu')(z)
    z = BatchNormalization()(z)
    residual3 = Conv1D(filters=256, kernel_size=1, padding='same')(input_acc) # 输入数据的维度需要与输出的维度相同
    z = Add()([z, residual3])
    # z = Bidirectional(LSTM(256, return_sequences=True))(z)
    # z = Bidirectional(LSTM(256, return_sequences=True))(z)
    # z = LSTM(64, return_sequences=True)(z)
    # z = LSTM(64, return_sequences=True)(z)
    #z = Model(inputs=input_acc, outputs=z)

    #for dilation_rate in dilation_rates:
    q = Conv1D(filters=32,
                kernel_size=3,
                padding='same',
                activation='relu')(input_w)
    q = Conv1D(filters=64,
                kernel_size=3,
                padding='same',
                activation='relu')(q)
    q = Conv1D(filters=128,
                kernel_size=3,
                padding='same',
                activation='relu')(q)
    q = Conv1D(filters=256,
                kernel_size=3,
                padding='same',
                activation='relu')(q)
    q = BatchNormalization()(q)
    residual4 = Conv1D(filters=256, kernel_size=1, padding='same')(input_w) # 输入数据的维度需要与输出的维度相同
    q = Add()([q, residual4])
    # q = Bidirectional(LSTM(256, return_sequences=True))(q)
    # q = Bidirectional(LSTM(256, return_sequences=True))(q)
    # q = LSTM(64, return_sequences=True)(q)
    # q = LSTM(64, return_sequences=True)(q)
    #q = Model(inputs=input_w, outputs=q)
    #x = SeqSelfAttention(attention_activation='sigmoid')(x)
    #x = Lambda(lambda x: x[:, 20 - 1, :])(x)  # select the final step

    # 对序列进行填充，使它们具有相同的时间步数
    # timesteps1 = x.shape[1]
    # timesteps2 = y.shape[1]
    # timesteps3 = z.shape[1]
    # timesteps4 = q.shape[1]

    # 找到最大的时间步数
    # max_timesteps = max(timesteps1, timesteps2, timesteps3, timesteps4)

    # output_padded_1 = ZeroPadding1D(padding=(0, max_timesteps - timesteps1))(x)
    # output_padded_2 = ZeroPadding1D(padding=(0, max_timesteps - timesteps2))(y)
    # output_padded_3 = ZeroPadding1D(padding=(0, max_timesteps - timesteps3))(z)
    # output_padded_4 = ZeroPadding1D(padding=(0, max_timesteps - timesteps4))(q)
    # 在需要设置断点的位置调用pdb.set_trace()
    #pdb.set_trace()

    # 在此处进行训练代码的编写和执行
    # 当程序执行到pdb.set_trace()时，会进入pdb调试模式，您可以在此处进行调试操作

    # 调试模式下的命令示例：
    # n：执行下一行代码
    # s：进入函数内部调试
    # q：退出调试模式
    # p <variable>：打印变量的值
    # ...

    # 调试完成后，继续执行剩余代码
    #print(x.shape)
    #print(lstm1_output_padded.shape)


    # combined = concatenate([output_padded_1, output_padded_2, output_padded_3, output_padded_4])
    combined = concatenate([x, y, z, q])


    combined = LSTM(256, return_sequences=True)(combined)
    combined = LSTM(256, return_sequences=False)(combined)

    # attention = Dense(1, activation='tanh')(combined)
    # attention = Activation('softmax')(attention)
    # weighted_output = Multiply(name='my_attention_weights')([combined, attention])
    # attention_output = Lambda(lambda x: K.sum(x, axis=1))(weighted_output)
    # print(attention_output.shape)
    #x = Flatten()(x)
    qq = Dense(128,activation='relu')(combined)
    qq = Dropout(0.5)(qq)
    qq = Dense(2)(qq)

    #def slice(x):
    #    return x[:,-2:,:]

    #pred_seq_train = Lambda(slice)(x)
    model = Model(inputs = [input_v, input_hpr, input_acc, input_w], outputs = qq)
    return model

model=build_model()
model.load_weights(model_name)

model.summary()

class MainPredict(object):
    def __init__(self):
        pass

    def pred(self,data):
        prediction = model.predict(data)
        print(prediction)
        print(prediction.shape)
        return prediction

#def main():
Predict = MainPredict()

start_time = time.time()
res = Predict.pred([X1_test_v,X1_test_hpr,X1_test_acc,X1_test_w])
end_time = time.time()

print(res)
testname = 'test_outputs' + str(epoch) + '.mat'
sio.savemat(testname, {'test_outputs': res})

#if __name__== '__main__':
#    main()

# attention_model = Model(inputs=model.input, outputs=model.get_layer('multiply').output)
# attention_weights = attention_model.predict([X1_test_v,X1_test_hpr,X1_test_acc,X1_test_w])
# attentionname = 'attention_weights' + str(epoch) + '.mat'
# sio.savemat(attentionname, {'attention_weights': attention_weights})

# attention_weights = attention_weights[0]

# plt.figure(dpi=120)
# # 绘制注意力热图
# plt.imshow(attention_weights, cmap=plt.get_cmap('Greens_r'), interpolation='nearest',aspect='auto')
# # 添加颜色条
# plt.colorbar()
# # 添加刻度标签
# plt.xticks(range(attention_weights.shape[1]))
# plt.yticks(range(attention_weights.shape[0]))
# # 显示图像
# plt.show()