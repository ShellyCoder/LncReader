import argparse
import sys
import os
import re

from Features.Generate import generateNumericSample
from pandas.core.frame import DataFrame
from step3_getScore import *


def fasta2dict(inf):
    myname = ""
    dict = {}
    for line in inf:
        line = line.strip()
        if line.startswith('>'):
            myname = line
            dict[myname] = ''
        else:
            dict[myname] += line
    return dict


def main(input_file, output_file):
    sys.path.append('./Features/')

    seq_dic = fasta2dict(open(input_file, 'r'))

    if not seq_dic:
        print("No sequences found in the file.")
        sys.exit(1)

    print(f"Total sequences read: {len(seq_dic)}")

    for name, sequence in seq_dic.items():
        if not name.startswith('>'):
            print(f"Invalid fasta header: {name}")
            sys.exit(1)
        if not sequence:
            print(f"Sequence for {name} is empty.")
            sys.exit(1)
        if re.search('[^ATCGatcg]', sequence):
            print(f"Sequence for {name} contains invalid characters.")
            sys.exit(1)

    print("All sequences read successfully.")

    with open(output_file, 'w') as fw:
        for key, value in seq_dic.items():
            input_feature = generateNumericSample(seq=value)
            a = [input_feature]
            data = DataFrame(a)
            data.columns = ['ORF.Max.Len', 'ORF.Max.Cov', 'hexamer_orf_step1', 'hexamer_orf_step3',
                            'hexamer_Full_length_step1',
                            'hexamer_Full_length_step3', 'FickettScore_orf', 'FickettScore_fulllength', 'freq_A',
                            'freq_T', 'freq_G',
                            'freq_C', 'AT_trans', 'AG_trans', 'AC_trans', 'TG_trans', 'TC_trans', 'GC_trans', 'A0_dis',
                            'A1_dis',
                            'A2_dis', 'A3_dis', 'A4_dis', 'T0_dis', 'T1_dis', 'T2_dis', 'T3_dis', 'T4_dis', 'G0_dis',
                            'G1_dis',
                            'G2_dis', 'G3_dis', 'G4_dis', 'C0_dis', 'C1_dis', 'C2_dis', 'C3_dis', 'C4_dis', 'pI_orf',
                            'pI_full_length',
                            'Signal.Peak', 'Average.Power', 'SNR', '0%', '25%', '50%', '75%', '100%']

            score = getAttention_Score(all_data=data, ndim=48)
            score_value = score[0] if score else None

            fw.write(key + '\t' + str(score_value) + '\t' + value + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process fasta files.')
    parser.add_argument('input_file', type=str, help='Input fasta file')
    parser.add_argument('output_file', type=str, help='Output result file')

    args = parser.parse_args()
    main(args.input_file, args.output_file)
