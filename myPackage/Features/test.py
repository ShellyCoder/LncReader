import sys
import os
sys.path.append('/work/data1/liutianyuan/pythonProject/python/lncReader/')
from Features.Generate import generateNumericSample
from pandas.core.frame import DataFrame
from step3_getScore import *
from Bio import SeqIO
import pandas as pd

sys.path.append('..')
print(os.getcwd())

# 输入你的 fasta 文件的路径
fasta_file_path = "/work/data1/liutianyuan/codingLncRNA/cncRNAdb_fasta/output_1.fasta"

# 读取fasta文件
records = list(SeqIO.parse(fasta_file_path, "fasta"))

# 用于存储所有的特征
all_features = []
all_ids = []

# 遍历每一个记录
for record in records:
    # 提取序列
    seq = str(record.seq)
    seq_id = record.id
    print(seq_id)
    # 计算特征
    input_feature = generateNumericSample(seq=seq)
    # 加入到列表中
    all_features.append(input_feature)
    all_ids.append(seq_id)


# input_seq = "CAGGATTTGTTTTATCCAACTCATCCCTTGACATCGTGCTTCACGATACATACTATGTAGTAGCCCATTTCCACTATGTTCTATCAATGGGAGCAGTGTTTGCTATCATAGCAGGATTTGTTCACTGATTCCCATTATTTTCAGGCTTCACCCTAGATGACACATGAGCAAAAGCCCACTTCGCCATCATATTCGTAGGAGTAAACATAACATTCTTCCCTCAACATTTCCTGGGCCTTTCAGGAATACCACGACGCTACTCAGACTACCCAGATGCTTACACCACATGAAACACTGTCTCTTCTATAGGATCATTTATTTCACTAACAGCTGTTCTCATCATGATCTTTATAATTTGAGAGGCCTTTGCTTCAAAACGAGAAGTAATATCAGTATCGTATGCTTCAACAAATTTAGAATGACTTCATGGCTGCCCTCCACCATATCACACATTCGAGGAACCAACCTATGTAAAAGTAAAATAAGAAAGGAAGGAATCGAACCCCCTAAAATTGGTTTCAAGCCAATCTCATATCCTATATGTCTTTCTCAATAAGATATTAGTAAAATCAATTACATAACTTTGTCAAAGTTAAATTATAGATCAATAATCTATATATCTTATATGGCCTACCCATTCCAACTTGGTCTACAAGACGCCACATCCCCTATTATAGAAGAGCTAATAAATTTCCATGATCACACACTAATAATTGTTTTCCTAATTAGCTCCTTAGTCCTCTATATCATCTCGCTAATATTAACAACAAAACTAACACATACAAGCACAATAGATGCACAAGAAGTTGAAACCATTTGAACTATTCTACCAGCTGTAATCCTTATCATAATTGCTCTCCCCTCTCTACGCATTCTATATATAATAGACGAAATCAACAACCCCGTATTAACCGTTAAAACCATAGGGCACCAATGATACTGAAGCTACGAATATACTGACTATGAAGACCTATGCTTTGATTCATATATAATCCCAACAAACGACCTAAAACCTGGTGAACTACGACTGCTAGAAGTTGATAACCGAGTCGTTCTGCCAATAGAACTTCCAATCCGTATATTAATTTCATCTGAAGACGTCCTCCACTCATGAGCAGTCCCCTCCCTAGGACTTAAAACTGATGCCATCCCAGGCCGACTAAATCAAGCAACAGTAACATCAAACCGACCAGGGTTATTCTATGGCCAATGCTCTGAAATTTGTGGATCTAACCATAGCTTTATGCCCATTGTCCTAGAAATGGTTCCACTAAAATATTTCGAAAACTGATCTGCTTCAATAATTTAATTTCACAAAAAAAAAAAAAAAAAAC"
# input_feature = generateNumericSample(seq=input_seq)
# print(input_feature)

# a=[input_feature]#包含两个不同的子列表[1,2,3,4]和[5,6,7,8]
# data=DataFrame(a)#这时候是以行为标准写入的

data=DataFrame(all_features)
data.columns = ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3', 'hexamer_Full_length_step1',
             'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A', 'freq_T', 'freq_G',
             'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis', 'A1_dis',
             'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis', 'G1_dis',
             'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf', 'pI_full_length',
             'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%']

score = getAttention_Score(all_data=data, ndim=48)

print(score)
df = pd.DataFrame({
    'ID': all_ids,
    'Score': score
})

# 导出为txt文件，使用tab作为列分隔符
df.to_csv('output_scores.txt', sep='\t', index=False)
