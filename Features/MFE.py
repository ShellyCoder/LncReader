import os
import subprocess
from Features.BaseClass import BaseClass
import re

class MFE(BaseClass):

    def __init__(self, rna_exe_path = "./MEF_RNAfold/RNAfold.exe"):
        name = os.name
        if name == "nt":
            curPath = str(os.path.realpath(__file__)).split("\\")[0:-1]
            curPath[0] = curPath[0] + "\\"
        else:
            curPath = str(os.path.realpath(__file__)).split("\\")[0:-1]
        self.cur_path = os.path.join(*curPath)
        self.rna_fold = rna_exe_path

    def calculation(self, seq, *args, **kwargs):
        with open(os.path.join(self.cur_path, "MEF_RNAfold", "temp.txt"), mode="w") as wh:
            wh.write(seq)
        p = subprocess.Popen(os.path.join(self.cur_path, "MEF_RNAfold", "RNAfold.exe") +
                             " " + os.path.join(self.cur_path, "MEF_RNAfold", "temp.txt"), stdout=subprocess.PIPE,
                             universal_newlines=True)
        reslut_lines = p.stdout
        final_line = None
        for line in reslut_lines:
            final_line = line
        p.kill()
        os.remove(os.path.join(self.cur_path, "MEF_RNAfold", "temp.txt"))
        regex_pattern = re.compile("\([+\-\s]*?\d+\.\d+\)")
        res = regex_pattern.findall(final_line)
        return float(res[0][1:-1])









