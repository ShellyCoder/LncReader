from typing import List
from Features.ORF_Length import ORFLength
from Features.FrameKmer import Kmer
from Features.Fickett import Fickett
from Features.CTD import CTD
from Features.ProtParam import ProteinPI
from Features.EIIP import EIIP
from Features.MFE import MFE
import os

def generateNumericSample(seq:str) -> List:
    orf_len_obj = ORFLength()
    orf_max_len, orf_max_cov, _ = orf_len_obj.calculation(seq)
    ### linux/windows/java, posix , nt , java
    name = os.name
    if name == "nt":
        curPath = str(os.path.realpath(__file__)).split("\\")[0:-1]
        curPath[0] = curPath[0] + "\\"
    else:
        curPath = str(os.path.realpath(__file__)).split("\\")[0:-1]
    fKmer_orf_s1 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "DNACodingDist", "orf_step1.txt" ), step_size=1)
    fKmer_orf_s3 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "DNACodingDist", "orf_step3.txt" ), step_size=3)
    fKmer_full_s1 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "DNACodingDist", "Full_step1.txt" ), step_size=1)
    fKmer_full_s3 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "DNACodingDist", "Full_step1.txt" ), step_size=3)
    hexamer_orf_step1 = fKmer_orf_s1.calculation(seq, if_orf=True)
    hexamer_orf_step3 = fKmer_orf_s3.calculation(seq, if_orf=True)
    hexamer_Full_length_step1 = fKmer_full_s1.calculation(seq, if_orf=False)
    hexamer_Full_length_step3 = fKmer_full_s3.calculation(seq, if_orf=False)
    fickett = Fickett()
    fick_orf = fickett.calculation(seq, if_orf=True)
    fick_full = fickett.calculation(seq, if_orf=False)
    ctd = CTD()
    ctd_results = ctd.calculation(seq)
    protein_pi = ProteinPI()
    pi_orf = protein_pi.calculation(seq, if_orf=True)
    pi_full = protein_pi.calculation(seq, if_orf=False)
    eiip = EIIP()
    eiip_res = eiip.calculation(seq)
    mfe = MFE()
    mfe_res = mfe.calculation(seq)
    results = [orf_max_len, orf_max_cov,
           hexamer_orf_step1, hexamer_orf_step3, hexamer_Full_length_step1, hexamer_Full_length_step3,
           fick_orf, fick_full] + list(ctd_results) + [pi_orf[1], pi_full[1]] + list(eiip_res) + [mfe_res]
    return results


