from model import transformer
from DataSet import GetRNAData
from DataSet import RNADataSet
from DataSet import RandomSelectBalancedTrainSamples_down
from DataSet import RandomSelectBalancedTrainSamples_up
from DataSet import GetGoldData
from WarmUpScheduler import GradualWarmupScheduler

'''
    第三方的模块
'''
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
from model import Forward_network
import joblib


def getAttention_Score(all_data, ndim=49):
    device = "cpu"
    # all_data = pd.read_table("feature.txt", header=0, encoding="utf-8", sep='\t',
    #                          index_col=0)

    if ndim == 49:
        print("Use three type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%', 'MFE']]

        trainedModel = transformer.ALBERT(in_features=1, d_model=256, heads=8, sequence_len=49, num_labels=1,
                                          cross_layers=2, parallel_Transformers=4, total_reuse_times=1, drop_p=0.35).to(device)
        trainedModel.load_state_dict(torch.load("./Weight/transformer_weight_final_seq_EIIP_MFE/Model_noise_0.pth",
                                                map_location=torch.device(device)))
        mean = np.load("./Data/mean_seq_EIIP_MFE.npy")
        std = np.load("./Data/std_seq_EIIP_MFE.npy")

    elif ndim == 48:
        print("Use two type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%']]

        trainedModel = transformer.ALBERT(in_features=1, d_model=256, heads=8, sequence_len=48, num_labels=1,
                                          cross_layers=2, parallel_Transformers=4, total_reuse_times=1, drop_p=0.35).to(device)
        trainedModel.load_state_dict(torch.load("./Weight/transformer_weight_final_seq_EIIP/Model_noise_0.pth",
                                                map_location=torch.device(device)))
        mean = np.load("./Data/mean_seq_EIIP.npy")
        std = np.load("./Data/std_seq_EIIP.npy")

    elif ndim == 38:
        print("Use one type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis']]

        trainedModel = transformer.ALBERT(in_features=1, d_model=256, heads=8, sequence_len=38, num_labels=1,
                                          cross_layers=2, parallel_Transformers=4, total_reuse_times=1, drop_p=0.35).to(device)
        trainedModel.load_state_dict(torch.load("./Weight/transformer_weight_final_seq/Model_noise_0.pth",
                                                map_location=torch.device(device)))

        mean = np.load("./Data/mean_seq.npy")
        std = np.load("./Data/std_seq.npy")

    # stop drop 和 batchNormalize
    trainedModel = trainedModel.eval()

    # preprocess data
    goldDataSet = RNADataSet(np.array(all_data), np.zeros(all_data.shape[0]), [], mean, std)
    goldLoader = DataLoader(goldDataSet, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=0)

    myGoldScore = []
    with torch.no_grad():
        for batch_idx, (BatchL, LabelsL) in enumerate(goldLoader):
            '''
                transformer model
            '''
            myInput = torch.unsqueeze(BatchL.float(), dim=-1).to(device)
            myOutput = torch.sigmoid(trainedModel(myInput)).squeeze().detach().cpu().numpy()
            myGoldScore.append(myOutput)

    return myGoldScore

def getDNN_Score(all_data, ndim=49):
    device = "cpu"
    # all_data = pd.read_table("feature.txt", header=0, encoding="utf-8", sep='\t',
    #                          index_col=0)

    if ndim == 49:
        print("Use three type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%', 'MFE']]

        trainedModel = Forward_network.MLE(in_channels=49, num_classes=1).to(device)
        trainedModel.load_state_dict(torch.load("./Weight/feedforward_weight_final_seq_EIIP_MFE/Model_epoch24.pth",
                                                map_location=torch.device(device)))
        mean = np.load("./Data/mean_seq_EIIP_MFE.npy")
        std = np.load("./Data/std_seq_EIIP_MFE.npy")


    elif ndim == 48:
        print("Use two type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%']]

        trainedModel = Forward_network.MLE(in_channels=48, num_classes=1).to(device)
        trainedModel.load_state_dict(torch.load("./Weight/feedforward_weight_final_seq_EIIP/Model_epoch24.pth",
                                                map_location=torch.device(device)))
        mean = np.load("./Data/mean_seq_EIIP.npy")
        std = np.load("./Data/std_seq_EIIP.npy")


    elif ndim == 38:
        print("Use one type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis']]

        trainedModel = Forward_network.MLE(in_channels=38, num_classes=1).to(device)
        trainedModel.load_state_dict(torch.load("./Weight/feedforward_weight_final_seq/Model_epoch24.pth",
                                                map_location=torch.device(device)))

        mean = np.load("./Data/mean_seq.npy")
        std = np.load("./Data/std_seq.npy")

    # stop drop 和 batchNormalize
    trainedModel = trainedModel.eval()

    # preprocess data
    goldDataSet = RNADataSet(np.array(all_data), np.zeros(all_data.shape[0]), [], mean, std)
    goldLoader = DataLoader(goldDataSet, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=0)

    myGoldScore = []
    with torch.no_grad():
        for batch_idx, (BatchL, LabelsL) in enumerate(goldLoader):

            '''
                Feedforward model
            '''
            myInput = BatchL.float().to(device)
            myOutput = torch.sigmoid(trainedModel(myInput)).squeeze().detach().cpu().numpy()
            myGoldScore.append(myOutput)

    return myGoldScore

def getSVM_Score(all_data, ndim=49):

    if ndim == 49:
        print("Use three type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%', 'MFE']]

        clf = joblib.load("./Weight/SVM_weight_final_seq_EIIP_MFE/Model_fold_final.model")

    elif ndim == 48:
        print("Use two type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%']]

        clf = joblib.load("./Weight/SVM_weight_final_seq_EIIP/Model_fold_final.model")

    elif ndim == 38:
        print("Use one type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis']]

        clf = joblib.load("./Weight/SVM_weight_final_seq/Model_fold_final.model")

    score = clf.predict_proba(all_data)

    score = score[:, 1].tolist()

    return score

def getBRF_Score(all_data, ndim=49):
    if ndim == 49:
        print("Use three type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%', 'MFE']]

        clf = joblib.load("./Weight/BRF_weight_final_seq_EIIP_MFE_2022_04_07/Model_fold_final.model")

    elif ndim == 48:
        print("Use two type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%']]

        clf = joblib.load("./Weight/BRF_weight_final_seq_EIIP_2022_04_07/Model_fold_final.model")

    elif ndim == 38:
        print("Use one type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis']]

        clf = joblib.load("./Weight/BRF_weight_final_seq_2022_04_07/Model_fold_final.model")

    score = clf.predict_proba(all_data)
    score = score[:, 1].tolist()

    return score

def getlogisitic_Score(all_data, ndim=49):
    if ndim == 49:
        print("Use three type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%', 'MFE']]

        clf = joblib.load("./Weight/logisitic_weight_final_seq_EIIP_MFE_2022_04_07/Model_fold_final.model")

    elif ndim == 48:
        print("Use two type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%']]

        clf = joblib.load("./Weight/logisitic_weight_final_seq_EIIP_MFE_2022_04_07/Model_fold_final.model")

    elif ndim == 38:
        print("Use one type of feature categories...")
        all_data = all_data[
            ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis']]

        clf = joblib.load("./Weight/logisitic_weight_final_seq_MFE_2022_04_07/Model_fold_final.model")

    score = clf.predict_proba(all_data)
    score = score[:, 1].tolist()

    return score


