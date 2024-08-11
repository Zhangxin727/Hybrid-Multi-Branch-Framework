#from keras.models import Model, load_model, Sequential
#from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, Concatenate
#from keras.optimizers import Adam
from unittest import result
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
tf.keras.backend.clear_session()

train = 'train_AUV_multihead.mat'
train_dataset = sio.loadmat(train)
train_data_v = train_dataset['train_data_v']
train_data_hpr = train_dataset['train_data_hpr']
train_data_acc = train_dataset['train_data_acc']
train_data_w = train_dataset['train_data_w']

print(train_data_v.shape)

train_label = train_dataset['train_label'].reshape([-1, 2])
train_data_len = train_label.shape[0]
print(train_data_len)

test = 'test_AUV_multihead_valid.mat'
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
Y1_train = train_label.reshape(-1,2)
Y1_test = test_label.reshape(-1,2)

mean = np.mean(X1_train_v, axis=0)  # 均值
X1_train_v -= mean  # 训练集
X1_test_v -= mean  # 测试集
std = np.std(X1_train_v, axis=0) # 标准差
X1_train_v /= std  # 训练集
X1_test_v /= std  # 测试集

# # 计算最小值和最大值
# min_val = np.min(X1_train_v, axis=0)
# max_val = np.max(X1_train_v, axis=0)
# # 最大最小值归一化
# X1_train_v = (X1_train_v - min_val) / (max_val - min_val)
# X1_test_v = (X1_test_v - min_val) / (max_val - min_val)

mean = np.mean(X1_train_hpr, axis=0)  # 均值
X1_train_hpr -= mean  # 训练集
X1_test_hpr -= mean  # 测试集
std = np.std(X1_train_hpr, axis=0) # 标准差
X1_train_hpr /= std  # 训练集
X1_test_hpr /= std  # 测试集

# 计算最小值和最大值
# min_val = np.min(X1_train_hpr, axis=0)
# max_val = np.max(X1_train_hpr, axis=0)
# # 最大最小值归一化
# X1_train_hpr = (X1_train_hpr - min_val) / (max_val - min_val)
# X1_test_hpr = (X1_test_hpr - min_val) / (max_val - min_val)

mean = np.mean(X1_train_acc, axis=0)  # 均值
X1_train_acc -= mean  # 训练集
X1_test_acc -= mean  # 测试集
std = np.std(X1_train_acc, axis=0) # 标准差
X1_train_acc /= std  # 训练集
X1_test_acc /= std  # 测试集

# # 计算最小值和最大值
# min_val = np.min(X1_train_acc, axis=0)
# max_val = np.max(X1_train_acc, axis=0)
# # 最大最小值归一化
# X1_train_acc = (X1_train_acc - min_val) / (max_val - min_val)
# X1_test_acc = (X1_test_acc - min_val) / (max_val - min_val)

mean = np.mean(X1_train_w, axis=0)  # 均值
X1_train_w -= mean  # 训练集
X1_test_w -= mean  # 测试集
std = np.std(X1_train_w, axis=0) # 标准差
X1_train_w /= std  # 训练集
X1_test_w /= std  # 测试集

# # 计算最小值和最大值
# min_val = np.min(X1_train_w, axis=0)
# max_val = np.max(X1_train_w, axis=0)
# # 最大最小值归一化
# X1_train_w = (X1_train_w - min_val) / (max_val - min_val)
# X1_test_w = (X1_test_w - min_val) / (max_val - min_val)
# #model = Sequential()

n_filters = 64
filter_width = 3
input_v = Input(shape=(4,3))
input_hpr = Input(shape=(20,3))
input_acc = Input(shape=(20,3))
input_w = Input(shape=(20,3))
#x = history_seq

model_name = 'NavNet'

load_previous_models = True
if os.path.exists('{}.h5'.format(model_name)):
    print('Load Previous Models')
    model = load_model(model_name+'.h5')

if not os.path.exists('{}.h5'.format(model_name)):

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

model.summary()

batch_size = 2**8
epochs = 300
checkpointer = ModelCheckpoint(os.path.join('model_{epoch:03d}.hdf5'),
                                   verbose=1, save_weights_only=True, period=10)

model.compile(Adam(), loss='mean_squared_error')
history =model.fit([X1_train_v,X1_train_hpr,X1_train_acc,X1_train_w], Y1_train,
                    batch_size=batch_size,
                    epochs=epochs,
                   callbacks=[checkpointer],
                   validation_data=([X1_test_v,X1_test_hpr,X1_test_acc,X1_test_w],Y1_test))
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Loss Over Time')
plt.show()

pre_result = model.predict([X1_test_v,X1_test_hpr,X1_test_acc,X1_test_w])
print(pre_result)
print(pre_result.shape)

#weight = model2.predict(X1_test(1,20,12))
#print(weight)
#sio.savemat('weight.mat', {'weight': weight})

sio.savemat('test_outputs.mat', {'test_outputs': pre_result})
sio.savemat('train_loss.mat', {'train_loss': history.history['loss']})
sio.savemat('val_loss.mat', {'val_loss': history.history['val_loss']})

#model.save(model_name + '.h5')
#model.save_weights(model_name + '.h5')


# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 4, 3)]       0           []                               
                                                                                                  
#  conv1d (Conv1D)                (None, 4, 32)        320         ['input_1[0][0]']                
                                                                                                  
#  input_2 (InputLayer)           [(None, 20, 3)]      0           []                               
                                                                                                  
#  input_3 (InputLayer)           [(None, 20, 3)]      0           []                               
                                                                                                  
#  input_4 (InputLayer)           [(None, 20, 3)]      0           []                               
                                                                                                  
#  conv1d_1 (Conv1D)              (None, 4, 64)        6208        ['conv1d[0][0]']                 
                                                                                                  
#  conv1d_5 (Conv1D)              (None, 20, 32)       320         ['input_2[0][0]']                
                                                                                                  
#  conv1d_10 (Conv1D)             (None, 20, 32)       320         ['input_3[0][0]']                
                                                                                                  
#  conv1d_15 (Conv1D)             (None, 20, 32)       320         ['input_4[0][0]']                
                                                                                                  
#  conv1d_2 (Conv1D)              (None, 4, 128)       24704       ['conv1d_1[0][0]']               
                                                                                                  
#  conv1d_6 (Conv1D)              (None, 20, 64)       6208        ['conv1d_5[0][0]']               
                                                                                                  
#  conv1d_11 (Conv1D)             (None, 20, 64)       6208        ['conv1d_10[0][0]']              
                                                                                                  
#  conv1d_16 (Conv1D)             (None, 20, 64)       6208        ['conv1d_15[0][0]']              
                                                                                                  
#  conv1d_3 (Conv1D)              (None, 4, 256)       98560       ['conv1d_2[0][0]']               
                                                                                                  
#  conv1d_7 (Conv1D)              (None, 20, 128)      24704       ['conv1d_6[0][0]']               
                                                                                                  
#  conv1d_12 (Conv1D)             (None, 20, 128)      24704       ['conv1d_11[0][0]']              
                                                                                                  
#  conv1d_17 (Conv1D)             (None, 20, 128)      24704       ['conv1d_16[0][0]']              
                                                                                                  
#  batch_normalization (BatchNorm  (None, 4, 256)      1024        ['conv1d_3[0][0]']               
#  alization)                                                                                       
                                                                                                  
#  conv1d_4 (Conv1D)              (None, 4, 256)       1024        ['input_1[0][0]']                
                                                                                                  
#  conv1d_8 (Conv1D)              (None, 20, 256)      98560       ['conv1d_7[0][0]']               
                                                                                                  
#  conv1d_13 (Conv1D)             (None, 20, 256)      98560       ['conv1d_12[0][0]']              
                                                                                                  
#  conv1d_18 (Conv1D)             (None, 20, 256)      98560       ['conv1d_17[0][0]']              
                                                                                                  
#  add (Add)                      (None, 4, 256)       0           ['batch_normalization[0][0]',    
#                                                                   'conv1d_4[0][0]']               
                                                                                                  
#  batch_normalization_1 (BatchNo  (None, 20, 256)     1024        ['conv1d_8[0][0]']               
#  rmalization)                                                                                     
                                                                                                  
#  conv1d_9 (Conv1D)              (None, 20, 256)      1024        ['input_2[0][0]']                
                                                                                                  
#  batch_normalization_2 (BatchNo  (None, 20, 256)     1024        ['conv1d_13[0][0]']              
#  rmalization)                                                                                     
                                                                                                  
#  conv1d_14 (Conv1D)             (None, 20, 256)      1024        ['input_3[0][0]']                
                                                                                                  
#  batch_normalization_3 (BatchNo  (None, 20, 256)     1024        ['conv1d_18[0][0]']              
#  rmalization)                                                                                     
                                                                                                  
#  conv1d_19 (Conv1D)             (None, 20, 256)      1024        ['input_4[0][0]']                
                                                                                                  
#  up_sampling1d (UpSampling1D)   (None, 20, 256)      0           ['add[0][0]']                    
                                                                                                  
#  add_1 (Add)                    (None, 20, 256)      0           ['batch_normalization_1[0][0]',  
#                                                                   'conv1d_9[0][0]']               
                                                                                                  
#  add_2 (Add)                    (None, 20, 256)      0           ['batch_normalization_2[0][0]',  
#                                                                   'conv1d_14[0][0]']              
                                                                                                  
#  add_3 (Add)                    (None, 20, 256)      0           ['batch_normalization_3[0][0]',  
#                                                                   'conv1d_19[0][0]']              
                                                                                                  
#  concatenate (Concatenate)      (None, 20, 1024)     0           ['up_sampling1d[0][0]',          
#                                                                   'add_1[0][0]',                  
#                                                                   'add_2[0][0]',                  
#                                                                   'add_3[0][0]']                  
                                                                                                  