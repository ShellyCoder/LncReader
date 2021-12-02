from Features.BaseClass import BaseClass
import Features.ORF as ORF

class ORFLength(BaseClass):

    def __init__(self, start_codons = "ATG", stop_codons = "TAG,TAA,TGA"):
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.Coverage = 0

    def __extract_feature_from_seq(self, seq, stt, stp):
        """extract features of sequence from fasta entry"""

        stt_coden = stt.strip().split(',')
        stp_coden = stp.strip().split(',')
        mRNA_seq = seq.upper()
        mRNA_size = len(seq)
        tmp = ORF.ExtractORF(mRNA_seq)
        (CDS_size1, CDS_integrity, CDS_seq1) = tmp.longest_ORF(start=stt_coden, stop=stp_coden)
        return mRNA_size, CDS_size1, CDS_integrity

    def calculation(self, seq, *args, **kwargs):
        seq = seq.upper()
        mRNA_size, CDS_size, CDS_integrity = self.__extract_feature_from_seq(seq=seq, stt=self.start_codons, stp=self.stop_codons)
        mRNA_len = mRNA_size
        CDS_len = CDS_size
        Coverage = float(CDS_len) / mRNA_len
        Integrity = CDS_integrity
        return CDS_len, Coverage, Integrity

