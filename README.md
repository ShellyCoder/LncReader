# LncReader

### Test Dataset Descriptions (stored in this repository)

#### 1. Mouse coding lncRNA Test Dataset
***File path***: `LncReader/test/Mouse_coding_lncRNA_test_dataset.fa`

This test dataset contains data specifically for mouse coding long non-coding RNAs (lncRNAs).

#### 2. Fruit Fly coding lncRNA Test Dataset
***File path***: `LncReader/test/Drosophila_coding_lncRNA_test_dataset.fa`

Similarly, this test dataset includes data specifically for fruit fly coding long non-coding RNAs (lncRNAs).

#### 3. Joint analysis of RNAseq-Ribosome-MS Test Dataset
***File path***: `LncReader/test/RNA-Ribosome-MS.xlsx`

This test dataset includes data specifically for leukaemia cell lines.

### Usage Guide
#### Input File
Prepare your input file, which should be in fasta format. For example, the file name could be `input.fasta`.

#### Output File
Define a name for your output file, which will be used to store the results processed by the script. For example, you might name your output file `output.txt`.

#### Instructions for Use

1. **Open the Command Line Interface**:
   Open a terminal (Linux or MacOS) or Command Prompt/PowerShell (Windows).

2. **Navigate to the Directory Containing the Script**:
   Use the `cd` command to navigate to the directory where your script is saved. For example:
   ```bash
   cd /path/to/script

3. **Run the Script**:
   Use the following command to run the script, replacing <input_file> and <output_file> with your actual file paths.
   ```bash
   python run_lncReader.py <input_file> <output_file>
   ```
   For example:
   ```bash
   python run_lncReader.py input.fasta output.txt
   ```

4. **Example Output**
   The output file will contain the identifier of each sequence, the calculated score (the probility of dual functional lncRNA), and the sequence itself. For example:
   ```python
   >Sequence1  0.4789213  GATTACAGATTACA...
   >Sequence2  0.5293812  GTCAGTCAGTCAGT...
    ...
   ```



-------------------------------------------------

We aim to explore whether LncReader provides a sophisticated and practical tool to identify dual functional lncRNAs and explore potentially lncRNA-encoded micropeptides, which might assist dissection the key roles of dual functional lncRNAs in either physiology or pathology conditions.

The research of LncReader is online. Please cite us with <a href="https://academic.oup.com/bib/article/24/1/bbac579/6961607" target="_blank">LncReader: identification of dual functional long noncoding RNAs using a multi-head self-attention mechanism</a>

<img width="865" alt="image" src="https://user-images.githubusercontent.com/49678387/231748358-30f19469-0167-4f76-b0a9-5f9ce69b6752.png">

