import numpy as np
from Features.BaseClass import BaseClass


class EIIP(BaseClass):

    def __init__(self,
                 percent_length = 0.25):
        self.percent_length = percent_length

    def calculation(self, seq: str, *args, **kwargs):
        seq = seq.upper()
        seqNumber = []
        for c in seq:
            if c == "A" : seqNumber.append(0.1260)
            elif c == "C": seqNumber.append(0.1340)
            elif c == "G": seqNumber.append(0.0806)
            elif c == "T": seqNumber.append(0.1335)

        seqNumber = np.array(seqNumber, dtype= np.float32)
        fourierSeq = np.fft.fft(seqNumber)
        seqSpectrum = np.abs(fourierSeq) ** 2.
        powerLength = len(seq)
        k = powerLength // 3
        dft = []
        if 0 <= k - 2 <= powerLength - 1: dft.append(seqSpectrum[k - 2])
        if 0 <= k - 1 <= powerLength - 1: dft.append(seqSpectrum[k - 1])
        if 0 <= k <= powerLength - 1: dft.append(seqSpectrum[k])
        if 0 <= k + 1 <= powerLength - 1: dft.append(seqSpectrum[k + 1])
        if 0 <= k + 2 <= powerLength - 1: dft.append(seqSpectrum[k + 2])
        if len(dft) != 0: dftMax = max(dft)
        else: dftMax = 0.
        signalPeak = dftMax
        averagePower = sum(seqSpectrum) / powerLength + 0.0
        sNR = dftMax / averagePower


        seqSpectrum = seqSpectrum[1: -1]
        orderPow = sorted(seqSpectrum, reverse=True)
        powPercent = orderPow[0: int(len(orderPow) * self.percent_length) + 1]
        quantile = np.quantile(powPercent, [0., 0.25, 0.5, 0.75, 1.])
        return signalPeak, averagePower, sNR, quantile[0], quantile[1], \
               quantile[2], quantile[3], quantile[4]





