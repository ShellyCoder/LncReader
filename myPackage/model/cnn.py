import tensorflow as tf
import pickle
import numpy as np
from keras.layers import Embedding
from keras import Input
from keras.layers import *
from keras.models import *

def get_cnn_model(x_train, y_train, embedding_matrix, kmer=3, MAX_SEQUENCE_LENGTH=4000, EMBEDDING_DIM = 100, word_index={}, BATCH_SIZE=64):
    # 构建CNN框架
    # Step1 使用 tf.data.Dataset.from_tensor_slices 进行加载
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # Step2 打乱数据(我们在getTrainData， getTestData时候已经打乱顺序，因此跳过)

    # Step3 预处理 (其实就是每个样本k-mer之后，印射到embedding后的向量)
    with open("../seq2feature/kmer_" + kmer + ".pickle", 'rb') as handle:
        embedding_dict = pickle.load(handle)
    embedding_matrix = np.zeros((len(embedding_dict) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        if word.upper() == 'NULL':
            pass
        else:
            embedding_vector = embedding_dict.get(word.upper())
            embedding_matrix[i] = embedding_vector

    '''
        word_index不需要+1，因为在制作Word_index已经考虑了null的情况，也就是本身word_index就已经维度+1了
        trainable=F 因为是用的是预训练好的模型
    '''
    embedding_layer = Embedding(len(word_index),
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    '''
        shape: (MAX_SEQUENCE_LENGTH, ) 预期输入的是多少长度的向量
        dtype: 数据类型
        batch_size: 一个批次是多少个样本
    '''
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', batch_size=None)
    embedded_sequences = embedding_layer(sequence_input)

    # 输入embedded_sequences起始就是三维变量 batch_size(64条或者指定条数的序列), steps(序列的每个碱基), feature(对每个碱基的表征)
    x = Conv1D(filters=64, kernel_size=5, activation="relu", padding='same', use_bias=True)(embedded_sequences)
    x = MaxPooling1D(pool_size=5, strides=None)(x)
    x = Conv1D(filters=64, kernel_size=5, activation="relu", padding='same', use_bias=True)(x)
    x = MaxPooling1D(pool_size=5, strides=None)(x)
    x = Conv1D(filters=64, kernel_size=5, activation="relu", padding='same', use_bias=True)(x)
    x = GlobalMaxPooling1D()(x) # 在steps维度（也就是第二维）对整个数据求最大值
    # x = Flatten()(x) # 需要Flatten嘛？

    # 添加两层全连接神经网络，dropout=0.5
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    # 输出层
    preds = Dense(2, activation='softmax')(x)
    model = Model(sequence_input, preds)
    return model



