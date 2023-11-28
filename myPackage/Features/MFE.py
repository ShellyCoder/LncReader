import subprocess
import re

def calculate_MFE(rna_sequence):
    # 使用 RNAfold 计算最小自由能
    process = subprocess.run(["/home/a2236013/miniconda3/envs/py39/bin/RNAfold"], input=rna_sequence, text=True, capture_output=True)
    output = process.stdout

    # 解析 RNAfold 的输出以获取最小自由能
    regex_pattern = re.compile("\([+\-\s]*?\d+\.\d+\)")
    match = regex_pattern.search(output)
    if match:
        mfe = float(match.group(0)[1:-1])
        return mfe

    return None  # 如果没有找到匹配项，返回 None
