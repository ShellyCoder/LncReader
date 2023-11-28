import math
from Features.BaseClass import BaseClass
from Features.ORF import ExtractORF

class Kmer(BaseClass):

    def __init__(self, coding_noncoding_prob_input_file, frame = 0, word_size=6, step_size=3):
        self.word_size = word_size
        self.step_size = step_size
        self.input_file = coding_noncoding_prob_input_file
        self.frame = frame

    def __coding_nocoding_potential(self, input_file):
        coding = {}
        noncoding = {}
        for line in open(input_file, mode="r").readlines():
            fields = line.strip("\n").split()
            coding[fields[0]] = float(fields[1])
            noncoding[fields[0]] = float(fields[2])
        return coding, noncoding

    def __word_generator(self, seq, word_size, step_size, frame = 0):
        """generate DNA word from sequence using word_size and step_size."""
        for i in range(frame, len(seq), step_size):
            word = seq[i:i + word_size]
            if len(word) == word_size:
                yield word

    def calculation(self, seq, *args, **kwargs):
        if len(seq) < self.word_size:
            return 0
        seq = seq.upper()
        if kwargs["if_orf"]:
            orf_extract_obj = ExtractORF(seq)
            _, _, seq = orf_extract_obj.longest_ORF()
        sum_of_log_ratio_0 = 0.0
        frame0_count = 0.0
        coding, noncoding = self.__coding_nocoding_potential(self.input_file)
        for k in self.__word_generator(seq=seq, word_size=self.word_size, step_size=self.step_size, frame=self.frame):
            if not coding.__contains__(k) or not noncoding.__contains__(k):
                continue
            if coding[k] > 0 and noncoding[k] > 0:
                sum_of_log_ratio_0 += math.log(coding[k] / noncoding[k])
            elif coding[k] > 0 and noncoding[k] == 0:
                sum_of_log_ratio_0 += 1
            elif coding[k] == 0 and noncoding[k] == 0:
                continue
            elif coding[k] == 0 and noncoding[k] > 0:
                sum_of_log_ratio_0 -= 1
            else:
                continue
            frame0_count += 1
        if frame0_count == 0:
            return -1
        else:
            return sum_of_log_ratio_0 / frame0_count

