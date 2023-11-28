import pandas
from torch.utils.data import Dataset
import torch
import numpy as np
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

'''
获取训练集和测试集
'''
def GetRNAData(cncCSVPath, lncCSVPath):
    # cncCSVPath = "../../data/myData/feature_file_from_gold_and_later/for_python_model_cncRNA_df.txt"
    # lncCSVPath = "../../data/myData/feature_file_from_gold_and_later/for_python_model_lncRNA_df.txt"
    # mRNACSVPath = "../../data/myData/feature_file_from_gold_and_later/for_python_model_mRNA_df.txt"
    '''
        CNC: 1, LNC: 0
    '''
    print("读入所有的数据，查看维度并检查是否有NA值")

    cncRNAOri = pandas.read_table(cncCSVPath, header=0, encoding="utf-8", sep='\t', index_col=0).dropna(axis=0)
    print("cnc nan {}".format(
        np.sum(np.array(np.isnan(np.array(pandas.DataFrame(cncRNAOri), dtype=np.float32)), dtype=np.float))))
    print("cnc shape: {}".format(cncRNAOri.shape))
    # print("type {}".format(cncRNAOri.dtypes))

    cncSampleNum = cncRNAOri.shape[0]
    cncLabels = np.ones(shape=[cncSampleNum])
    print("label 1 for cncRNA")
    # cncRNAOri.head(5)


    lncRNAOri = pandas.read_table(lncCSVPath, header=0, encoding="utf-8", sep='\t', index_col=0).dropna(axis=0)
    print("lncRNA nan {}".format(
        np.sum(np.array(np.isnan(np.array(pandas.DataFrame(lncRNAOri), dtype=np.float32)), dtype=np.float))))
    print("lncRNA shape: {}".format(lncRNAOri.shape))
    # print("type {}".format(lncRNAOri.dtypes))

    lncSampleNum = lncRNAOri.shape[0]
    lncLabels = np.zeros(shape=[lncSampleNum])
    print("label 0 for cncRNA")
    # lncRNAOri.head(5)

    combineData = pandas.concat([cncRNAOri, lncRNAOri], axis=0)
    combineLabels = np.concatenate([cncLabels, lncLabels], axis=0)
    combineSeq = []

    return combineData, combineLabels, combineSeq

'''
继承改写Dataset
'''
class RNADataSet(Dataset):
    '''
    继承Dataset类属性，改写初始化参数，getitem()和__len__()方法
    '''
    def __init__(self, numericData, labels, seqData, mean: np.array, std: np.array):
        super().__init__()
        assert numericData.shape[0] == labels.shape[0], "The numbers of data and labels are not same."
        self.numericData, self.labels, self.seqData = numericData, labels, seqData
        self.numericData = (self.numericData - mean) / std
        self.numericData = torch.from_numpy(numericData).float()
        self.labels = torch.from_numpy(self.labels).long()


    def __getitem__(self, item):
        return self.numericData[item], self.labels[item]

    def __len__(self):
        return self.numericData.shape[0]


'''
训练集中所有cncRNA样本和随机n倍的阴性集合
'''
def RandomSelectBalancedTrainSamples_down(data, label, cncFold=1, lncFold=1):
    minNum = min(Counter(label)[0], Counter(label)[1.0])

    cncIndices = np.array(list(np.where(label == 1.0))).reshape(1, -1).flatten()
    cncIndices_random = np.random.choice(cncIndices, size=minNum * cncFold, replace=False)

    lncIndices = np.array(list(np.where(label == 0.0))).reshape(1, -1).flatten()
    lncIndices_random = np.random.choice(lncIndices, size=minNum * lncFold, replace=False)
    # print(lncIndices_random)
    # print(label[lncIndices_random])

    balanceCnc = data[cncIndices_random,]
    cncLabels = label[cncIndices_random]
    print("cnc shape: {}".format(balanceCnc.shape))
    balanceLnc = data[lncIndices_random,]
    lncLabels = label[lncIndices_random]
    print("lnc shape: {}".format(balanceLnc.shape))

    print(Counter(np.concatenate([cncLabels, lncLabels], axis=0)))

    combineData = np.concatenate([balanceCnc, balanceLnc], axis=0)
    combineLabels = np.concatenate([cncLabels, lncLabels], axis=0)

    return combineData, combineLabels, []

def RandomSelectBalancedTrainSamples_up(data, label):
    maxNum = max(Counter(label)[0], Counter(label)[1.0])

    cncIndices = np.array(list(np.where(label == 1.0))).reshape(1, -1).flatten()
    cncIndices_random = np.random.choice(cncIndices, size=maxNum, replace=True)

    lncIndices = np.array(list(np.where(label == 0.0))).reshape(1, -1).flatten()
    lncIndices_random = np.random.choice(lncIndices, size=maxNum, replace=False)
    # print(lncIndices_random)
    # print(label[lncIndices_random])

    balanceCnc = data[cncIndices_random,]
    cncLabels = label[cncIndices_random]
    print("cnc shape: {}".format(balanceCnc.shape))
    balanceLnc = data[lncIndices_random,]
    lncLabels = label[lncIndices_random]
    print("lnc shape: {}".format(balanceLnc.shape))

    print(Counter(np.concatenate([cncLabels, lncLabels], axis=0)))

    combineData = np.concatenate([balanceCnc, balanceLnc], axis=0)
    combineLabels = np.concatenate([cncLabels, lncLabels], axis=0)

    return combineData, combineLabels, []


def GetGoldData(goldataCSVPath):
    '''
        CNC: 1, LNC: 0
    '''
    print("读入所有的数据，查看维度并检查是否有NA值")

    goldataRNAOri = pandas.read_table(goldataCSVPath, header=0, encoding="utf-8", sep='\t', index_col=0).dropna(axis=0)
    print("goldData nan {}".format(
        np.sum(np.array(np.isnan(np.array(pandas.DataFrame(goldataRNAOri), dtype=np.float32)), dtype=np.float))))
    print("goldData shape: {}".format(goldataRNAOri.shape))
    # print("type {}".format(cncRNAOri.dtypes))

    goldataLabels = np.array(goldataRNAOri["V5.validation"])

    # print("label 1 for cncRNA")
    # cncRNAOri.head(5)

    return goldataRNAOri, goldataLabels

def GetGoldData_nature(goldataCSVPath):
    '''
        CNC: 1, LNC: 0
    '''
    print("读入所有的数据，查看维度并检查是否有NA值")

    goldataRNAOri = pandas.read_table(goldataCSVPath, header=0, encoding="utf-8", sep='\t', index_col=0).dropna(axis=0)
    print("goldData nan {}".format(
        np.sum(np.array(np.isnan(np.array(pandas.DataFrame(goldataRNAOri), dtype=np.float32)), dtype=np.float))))
    print("goldData shape: {}".format(goldataRNAOri.shape))
    # print("type {}".format(cncRNAOri.dtypes))

    goldataLabels = np.array(np.zeros(goldataRNAOri.shape[0]))

    # print("label 1 for cncRNA")
    # cncRNAOri.head(5)

    return goldataRNAOri, goldataLabels


def plot_durations(y, title_list):

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    y = np.array(y)
    n_fig = y.shape[1] * 100 + 11
    plt.figure(y.shape[1], figsize=(4,16))
    plt.clf()

    for i in range(y.shape[1]):
        plt.subplot(n_fig + i)
        plt.title(title_list[i], color='blue')
        plt.plot(y[:, i], marker='o', markersize=6)

    plt.pause(0.01)  # pause a bit so that plots are updated

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


if __name__ == "__main__":
    numberic , labels , seq = GetRNAData("./Data/cncRNA.csv","./Data/lncRNA.csv")
    testDataset = RNADataSet(numberic, labels, seq)
    print(testDataset.__getitem__(5666))
