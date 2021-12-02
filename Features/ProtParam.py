import re
from Bio.Seq import Seq
from Features.ORF  import ExtractORF
from Bio.SeqUtils import ProtParam
from Features.BaseClass import BaseClass


class ProteinPI(BaseClass):

	def __init__(self):
		pass

	def __mRNA_translate(self, mRNA):
		n = len(mRNA)
		if n % 3 == 2: mRNA = mRNA[0:-2]
		elif n % 3 == 1: mRNA = mRNA[0:-1]
		return Seq(mRNA).translate()

	def __protein_param(self, putative_seqprot):
		return putative_seqprot.instability_index(), \
			   putative_seqprot.isoelectric_point(), \
			   putative_seqprot.gravy()

	def calculation(self, seq,  *args, **kwargs):
		seq = seq.upper()
		if_orf = kwargs["if_orf"]
		strinfoAmbiguous = re.compile("[XBZJU]", re.I)
		ptU = re.compile("U", re.I)
		seqRNA = ptU.sub("T", str(seq).strip())
		seqRNA = seqRNA.upper()
		if if_orf:
			CDS_size1, CDS_integrity, seqCDS = ExtractORF(seqRNA).longest_ORF(start=['ATG'], stop=['TAA', 'TAG', 'TGA'])
			seqprot = self.__mRNA_translate(seqCDS)
		else:
			seqprot = self.__mRNA_translate(seq)
		seqprot = str(seqprot).replace("*", "")
		pep_len = len(seqprot.strip("*"))
		newseqprot = strinfoAmbiguous.sub("", str(seqprot))
		protparam_obj = ProtParam.ProteinAnalysis(str(newseqprot.strip("*")))
		if pep_len > 0:
			Instability_index, PI, Gravy = self.__protein_param(protparam_obj)
		else:
			Instability_index = 0.0
			PI = 0.0
			Gravy = 0.0
		return Instability_index, PI, Gravy


