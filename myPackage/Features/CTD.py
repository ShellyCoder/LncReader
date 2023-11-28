from Features.BaseClass import BaseClass

class CTD(BaseClass):

    def __init__(self):
        pass

    def calculation(self, seq: str, *args, **kwargs) -> tuple:
        """
        :param seq: The sequence.
        :return: 28 seq features.
        """
        seq = seq.upper()
        n = len(seq) - 1
        n = float(n)
        num_A, num_T, num_G, num_C = 0, 0, 0, 0
        AT_trans, AG_trans, AC_trans, TG_trans, TC_trans, GC_trans = 0, 0, 0, 0, 0, 0
        for i in range(len(seq) - 1):
            if seq[i] == "A":
                num_A = num_A + 1
            if seq[i] == "T":
                num_T = num_T + 1
            if seq[i] == "G":
                num_G = num_G + 1
            if seq[i] == "C":
                num_C = num_C + 1
            if (seq[i] == "A" and seq[i + 1] == "T") or (seq[i] == "T" and seq[i + 1] == "A"):
                AT_trans = AT_trans + 1
            if (seq[i] == "A" and seq[i + 1] == "G") or (seq[i] == "G" and seq[i + 1] == "A"):
                AG_trans = AG_trans + 1
            if (seq[i] == "A" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "A"):
                AC_trans = AC_trans + 1
            if (seq[i] == "T" and seq[i + 1] == "G") or (seq[i] == "G" and seq[i + 1] == "T"):
                TG_trans = TG_trans + 1
            if (seq[i] == "T" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "T"):
                TC_trans = TC_trans + 1
            if (seq[i] == "G" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "G"):
                GC_trans = GC_trans + 1
        a, t, g, c = 0, 0, 0, 0
        A0_dis, A1_dis, A2_dis, A3_dis, A4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        T0_dis, T1_dis, T2_dis, T3_dis, T4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        G0_dis, G1_dis, G2_dis, G3_dis, G4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        C0_dis, C1_dis, C2_dis, C3_dis, C4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(len(seq) - 1):
            if seq[i] == "A":
                a = a + 1
                if a == 1:
                    A0_dis = ((i * 1.0) + 1) / n
                if a == int(round(num_A / 4.0)):
                    A1_dis = ((i * 1.0) + 1) / n
                if a == int(round(num_A / 2.0)):
                    A2_dis = ((i * 1.0) + 1) / n
                if a == int(round((num_A * 3 / 4.0))):
                    A3_dis = ((i * 1.0) + 1) / n
                if a == num_A:
                    A4_dis = ((i * 1.0) + 1) / n
            if seq[i] == "T":
                t = t + 1
                if t == 1:
                    T0_dis = ((i * 1.0) + 1) / n
                if t == int(round(num_T / 4.0)):
                    T1_dis = ((i * 1.0) + 1) / n
                if t == int(round((num_T / 2.0))):
                    T2_dis = ((i * 1.0) + 1) / n
                if t == int(round((num_T * 3 / 4.0))):
                    T3_dis = ((i * 1.0) + 1) / n
                if t == num_T:
                    T4_dis = ((i * 1.0) + 1) / n
            if seq[i] == "G":
                g = g + 1
                if g == 1:
                    G0_dis = ((i * 1.0) + 1) / n
                if g == int(round(num_G / 4.0)):
                    G1_dis = ((i * 1.0) + 1) / n
                if g == int(round(num_G / 2.0)):
                    G2_dis = ((i * 1.0) + 1) / n
                if g == int(round(num_G * 3 / 4.0)):
                    G3_dis = ((i * 1.0) + 1) / n
                if g == num_G:
                    G4_dis = ((i * 1.0) + 1) / n
            if seq[i] == "C":
                c = c + 1
                if c == 1:
                    C0_dis = ((i * 1.0) + 1) / n
                if c == int(round(num_C / 4.0)):
                    C1_dis = ((i * 1.0) + 1) / n
                if c == int(round(num_C / 2.0)):
                    C2_dis = ((i * 1.0) + 1) / n
                if c == int(round(num_C * 3 / 4.0)):
                    C3_dis = ((i * 1.0) + 1) / n
                if c == num_C:
                    C4_dis = ((i * 1.0) + 1) / n

        return num_A / n, num_T / n, num_G / n, num_C / n, \
               AT_trans / (n - 1), AG_trans / (n - 1), AC_trans / (n - 1), \
               TG_trans / (n - 1), TC_trans / (n - 1), GC_trans / (n - 1), \
               A0_dis, A1_dis, A2_dis, A3_dis, A4_dis, \
               T0_dis, T1_dis, T2_dis, T3_dis, T4_dis, \
               G0_dis, G1_dis, G2_dis, G3_dis, G4_dis, \
               C0_dis, C1_dis, C2_dis, C3_dis, C4_dis



