import math

from fairseq.data import Dictionary
import torch
import re
import struct
MASK = '<mas>'
PREBOS = '<pbos>'
EON='[EON]'

ArabicPattern=r'(\d)\-?(@@)?'
EnglishValue={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'first':1,'second':2,'forty':4,'fifty':5,'eigh':8,
              'thirty':30,'sixty':60,'third':3,'fourth':4,'fifth':5,'twelve':12,
              'twenty':20,'eighteen':18,'thirteen':13,'seventeen':17,'fifteen':15,'nin':9,'eighty':80,'fourteen':14,'sixteen':16,'sixth':6,'seventh':7,
              'eighth':8,'eleven':11}
EnglishUnit={
              'hundred':1e2,'thousand':1e3,'million':1e6,'billion':1e9,'trillion':1e12}

ChineseValue={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'零':0}
Arabic=list('0123456789')
ChineseUnit={'十':10,'百':100,'千':1000,'万':10000,'亿':100000000,'兆':1000000000000}

class NumeralDictionary(Dictionary):
	def __init__(
			self,
			pad='<pad>',
			eos='</s>',
			unk='<unk>',
			bos='<s>',
			extra_special_symbols=None,
	):
		super().__init__(bos=bos,pad=pad,eos=eos,unk=unk,extra_special_symbols=extra_special_symbols)
		sz=len(self.symbols)
		self.id2value=[0]*sz
def cn2an(s):
	if s.endswith('@@'):
		s=s[:-2]
	s=s[::-1]
	ans=0
	subunit=1
	base_unit=1
	while s[0] not in ChineseUnit and s[0] not in ChineseValue and s[0] not in Arabic:
		s=s[1:]
	for c in s:
		if c in ChineseUnit:
			uval=ChineseUnit[c]
			if uval>base_unit:
				unit=base_unit=uval
			else:
				unit=uval
		elif c in ChineseValue:
			ans+=ChineseValue[c]*subunit*base_unit
			subunit*=10
		elif c in Arabic:
			ans+=int(c)*subunit*base_unit
			subunit*=10
		elif c in '点.':
			ans/=subunit
			subunit=1
		elif c==',':
			continue
		elif c in'负-':
			ans=-ans
		else:
			return None
	if s[-1] in ChineseUnit:
		ans+=subunit*base_unit
	return ans

def en2an(s):
	if s.endswith('@@'):
		s=s[:-2]
	if s in EnglishValue:
		return EnglishValue[s]
	if s in EnglishUnit:
		return EnglishUnit[s]
	return None


class MultihotNumDictionary(Dictionary):
	def __init__(
			self,
			pad='<pad>',
			eos='</s>',
			unk='<unk>',
			bos='<s>',
			extra_special_symbols=None,
	):
		super().__init__(bos=bos,pad=pad,eos=eos,unk=unk,extra_special_symbols=extra_special_symbols)
		sz=len(self.symbols)
		self.id2vector=[torch.tensor([0]*33,dtype=torch.long)]*sz
	def build_multihot_vector(self):
		sz=len(self.symbols)
		self.id2vector = [torch.tensor([0] * 33, dtype=torch.long)] * sz
		cnnsym=list(ChineseValue.keys())+list(ChineseUnit.keys())+Arabic
		ennsym=list(EnglishValue.keys())+list(EnglishUnit.keys())
		for i,s in enumerate(self.symbols):
			# print(s)
			if set(cnnsym).intersection(set(s)):
				val=cn2an(s)
				if val is None:
					continue
			elif s in ennsym:
				val=en2an(s)
				if val is None:
					continue
				# print(s,val)
			else:
				continue
			bval=format(struct.unpack('!I',struct.pack('!f',val))[0],'032b')
			vec=torch.tensor([1]+list(map(lambda x:int(x),list(bval))),dtype=torch.long)

			self.id2vector[i]=vec
		self.id2vector=torch.stack(self.id2vector)
	def get_multihot_sequence(self,sent):
		sent_size=sent.size()
		res = torch.index_select(self.id2vector.to(sent.device),0,sent.view(-1))
		res=res.view(*sent_size,-1)
		return res
	def get_dic_map(self,dic):
		res=[]
		for s in self.symbols:
			res.append(dic.index(s))
		return torch.LongTensor(res)


class CombinedDictionary(Dictionary):
	def __init__(
        self,
		dict1,
		dict2,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        bos='<s>',
        extra_special_symbols=None,
    ):
		super().__init__(bos=bos,pad=pad,eos=eos,unk=unk,extra_special_symbols=extra_special_symbols)
		def build_dmap(d):
			dmap=[self.unk()]*len(d)
			for v in d.symbols:
				j=self.add_symbol(v)
				dmap[d.index(v)]=j
			return torch.LongTensor(dmap)
		self.d1map=build_dmap(dict1)
		self.d2map=build_dmap(dict2)
		def build_dbmap(d):
			dbmap=[d.pad()]*len(self)
			for v in d.symbols:
				dbmap[self.index(v)]=d.index(v)
			return torch.LongTensor(dbmap)
		self.d1bmap=build_dbmap(dict1)
		self.d2bmap=build_dbmap(dict2)
		self.dict1=dict1
		self.dict2=dict2
	def combine_probs(self,probs1,probs2,weights):
		assert not torch.isnan(probs1).any(),torch.isnan(probs1).nonzero()
		assert not torch.isinf(probs1).any(),torch.isinf(probs1).nonzero()
		# assert not torch.isnan(probs2).any(),torch.isnan(probs2).nonzero()
		# assert not torch.isinf(probs2).any(),torch.isinf(probs2).nonzero()
		d1map=self.d1map.to(probs1.device)
		d2map=self.d2map.to(probs2.device)
		bsz,seqlen=probs1.size()[:2]
		w0,w1=weights.unbind(dim=-1)
		probs1 = probs1 + (w0.unsqueeze(2))
		
		probs1=torch.where(torch.isinf(probs1),torch.full_like(probs1,-math.inf),probs1)
		# print(lprobs.siz())
		# num_probs[:,:,self.num_dict.eos()]=-math.inf
		probs2 = probs2 + (w1.unsqueeze(2))
		# TODO stop grads with math.inf
		probs2=torch.where(torch.isinf(probs2),torch.full_like(probs2,-math.inf),probs2)
		assert probs2.size(-1)==len(self.dict2)
		assert len(d2map)==probs2.size(-1)
		# assert not torch.isnan(probs2).any(), torch.isnan(probs2).nonzero()
		# assert not torch.isinf(probs2).any(), torch.isinf(probs2).nonzero()

		final_probs1 = probs1.new_zeros(bsz, seqlen, len(self)).fill_(-math.inf)
		final_probs2 = probs2.new_zeros(bsz, seqlen, len(self)).fill_(-math.inf)
		# print(probs1.size(),final_probs1.size(),d1map)
		final_probs1 = final_probs1.index_copy(-1, d1map, probs1)
		final_probs2 = final_probs2.index_copy(-1, d2map, probs2)
		bayasian_mask = final_probs2.gt(final_probs1)
		probs = torch.logaddexp(final_probs1,final_probs2)
		probs.type_as(probs1)
		probs = probs.squeeze(1)
		bayasian_mask=bayasian_mask.squeeze(1)
		return probs,bayasian_mask
	def map_d1(self,labels):
		sz=labels.size()
		d1map=self.d1map.to(labels.device)
		mapped_labels=torch.index_select(d1map,dim=0,index=labels.view(-1)).view(sz)
		return mapped_labels