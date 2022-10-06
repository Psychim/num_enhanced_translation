import torch
from decimal import Decimal
import torch.nn as nn
from .dictionary import EON
from fairseq.utils import set_incremental_state,get_incremental_state
from fairseq.incremental_decoding_utils import with_incremental_state
from .timer import TicTocTimer
from .patterns import *
import re
import math
thousand=1000
million=thousand*1000
billion=million*1000
trillion=billion*1000
hundred=100
ChineseValue={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'零':0,'两':2,'○':0}
Arabic=list('0123456789')
ChineseUnit={'十':10,'百':100,'千':1000,'万':10000,'亿':100000000,'兆':1000000000000}
IgnoreToken=[',',' ','.','多','份','第','起','分','秒','点','时']
StopToken=['日','号','个','月','年','月份']
ForbidToken=['几']
month_table=['january','february','march','april','may','june','july','august','september','october','november','december']

def token_is_num(token):
	if token.endswith('@@'):
		token=token[:-2]
	for c in token:
		if c in IgnoreToken:
			continue
		elif c in StopToken:
			continue
		elif c not in ChineseValue and c not in ChineseUnit and c not in Arabic:
			return False
	return True
def cn2an(s):
	if s.endswith('@@'):
		s=s[:-2]
	s=s[::-1]
	ans=Decimal('0')
	subunit=1
	base_unit=1
	digit_unit=1
	if not s:
		return None
	# print(s)
	for c in s:
		if c in ChineseUnit:
			uval=ChineseUnit[c]
			if uval>base_unit:
				if uval < base_unit * subunit*digit_unit:
					return None
				base_unit=uval
				subunit=1
				digit_unit=1
			else:
				subunit=uval
				digit_unit=1
		elif c in ChineseValue:
			ans+=ChineseValue[c]*subunit*base_unit*digit_unit
			digit_unit*=10
		elif c in Arabic:
			ans+=int(c)*subunit*base_unit*digit_unit
			digit_unit*=10
		elif c in '点.':
			ans/=digit_unit
			digit_unit=1
		elif c in IgnoreToken:
			continue
		elif c in StopToken:
			continue
		elif c in'负-':
			ans=-ans
		else:
			return None
	if s[-1] in ChineseUnit:
		ans+=digit_unit*base_unit*subunit
	return ans
def divide_en(num,unit,name):
	pre=an2en(num//unit)
	suf=an2en(num%unit)
	if unit==100 and suf!='':
		suf='and '+suf
	return pre+' '+name+' '+suf
def base_en(num):
	def standarlize(en):
		splits=[]
		for i in range(len(en),0,-3):
			splits.append(en[max(0,i-3):i])
		return ','.join(splits[::-1])
	if num==0:
		return ''
	if int(num)==num:
		en=str(int(num))
		en=standarlize(en)
	else:
		en=str(num)
		if '.' in en:
			idx=en.index('.')
			en=standarlize(en[:idx])+en[idx:]
	return en
def an2en(num):
	if int(num)!=num:
		return str(num)
	num=int(num)
	unit=''
	unit_val=1
	def float_digit(num):
		s=str(num)
		return len(s[s.index('.'):])
	if num/trillion>=1 and float_digit(num/trillion) <= 4:
		unit=' trillion'
		unit_val=trillion
	elif num/billion>=1  and float_digit(num/billion) <= 4:
		unit=' billion'
		unit_val=billion
	elif num/million>=1 and float_digit(num/million) <= 4:
		unit=' million'
		unit_val=million
	if len(str(num/unit_val))-float_digit(num/unit_val) <= 6:
		num/=unit_val
	else:
		unit=''
	return base_en(num)+unit
def cn2en(cn):
	an=cn2an(cn)
	if an is None:
		return ''
	en=an2en(an)
	return en
def ordinal2en(chinese_ord):
	# v=compiled_integer.search(chinese_ord)
	v=chinese_ord[1:]
	if not v:
		return ''
	v=cn2an(v)
	if v is None:
		return ''
	en=ordinal_an2en(v)
	return en
def ordinal_an2en(num):
	int_n=int(num)
	assert int_n==num
	str_n=str(int_n)
	suffix = ['st', 'nd', 'rd']
	if int(str_n[-1])>=1 and int(str_n[-1]) <= 3:
		s = str_n + suffix[int(str_n[-1]) - 1]
	else:
		s = str_n + 'th'
	return s
def date2en(chinese_date):
	# print(dates)
	def extract(pattern):
		v=re.search(pattern,chinese_date)
		if not v:
			return None,None
		v=compiled_integer.search(v.group(0))

		return cn2an(v.group(0)),v.group(0)

	year,year_str=extract(year_pattern)
	if year_str is not None and year_str[0] in '零0' and 0<=year and year<10:
		# print(year_str)
		year+=2000
	month,_=extract(month_pattern)
	day,_=extract(day_pattern)
	en=''
	if month is not None:
		month=month_table[int(month)-1]
		en=month
	if day is not None:
		# day=ordinal_an2en(day)
		day=an2en(day)
		if en:
			en+=' '
		en+=day
	if year is not None:
		year=str(year)
		if en:
			en+=' , '
		en+=year
	return en
def get_chinese_num_type(cn):
	if ('时' in cn or '点' in cn or '分' in cn or '秒' in cn) and \
			compiled_time.match(cn):
		return 'time'
	elif ('年' in cn or '月' in cn or '日' in cn or '号' in cn) and \
			compiled_date.match(cn):
		return 'date'
	elif '第' in cn and compiled_ordinal.match(cn):
		return 'ordinal'
	elif '成' in cn and compiled_percent.match(cn):
		return 'percent'
	elif '旬' in cn and compiled_age.match(cn):
    		return 'age'
	elif compiled_number.match(cn):
		return 'number'
	else:
		# if DEBUGGING:
		# 	print('unknown number',cn)
		assert True
		return ''
def percent2en(chinese_percent):
	# v=compiled_number.search(chinese_percent)
	v=chinese_percent[:-1]
	if not v:
		return ''
	# v=v.group(0)
	v=cn2an(v)
	if v is None:
		return ''
	v*=10
	en=an2en(v)+' %'
	return en
def time2en(chinese_time):
	def extract(pattern):
		v=re.search(pattern,chinese_time)
		if not v:
			return ''
		v=v.group(0)
		if v=='半':
			return '30'
		# v=compiled_integer.search(v)
		v=v[:-1]
		v=cn2en(v)
		return v
	hour=extract(hour_pattern)
	min=extract(minute_pattern)
	second=extract(second_pattern)
	en=[]
	if hour:
		en.append(hour)
	if min:
		if len(min)==1:
			min='0'+min
		en.append(min)
	else:
		if hour:
			en.append('00')
	if second:
		en.append(second)
	en=' : '.join(en)
	return en
def age2en(cn):
	age=compiled_integer.search(cn)
	if not age:
		return ''		
	age=cn2an(age.group(0))
	age*=10
	age=an2en(age)
	return age
def findall_nums(src_sents):
	bsz=len(src_sents)
	numsets = [[] for _ in range(bsz)]
	# 连接所有文本，只执行一次正则匹配以加速
	joined_sents = '\n' + '\n'.join(src_sents)
	line_index = [i for i, c in enumerate(joined_sents) if c == '\n']
	# print(joined_sents)
	matches = compiled_pattern.finditer(joined_sents)
	# print(src_sents[i])
	for m in matches:
		pos = m.start()
		# print(m.group())
		i = 0
		while i + 1 < len(line_index) and line_index[i + 1] < pos:
			i += 1
		assert m.group() in src_sents[i]
		assert m.start()!=m.end()
		numsets[i].append((m.group(), m.start() - line_index[i] - 1, m.end() - line_index[i] - 1))
	return numsets

@with_incremental_state
class RuleNumTranslator(nn.Module):
	def __init__(self,args,src_dict,num_dict):
		super().__init__()
		self.args=args
		self.src_dict=src_dict
		self.num_dict=num_dict
		self.timers=[TicTocTimer() for _ in range(3)]
	def forward(self,src_tokens,attn,incremental_state=None,step=0,requires_label=False,**extra_args):
		def build_src_sentence(src_token):
			sent=''
			position_map=[]
			i=0
			for idx in src_token:
				position_map.append(i)
				if idx==0 or idx==2:
					continue
				c = self.src_dict[idx]
				if c.endswith('@@'):
					c=c[:-2]
				# else:
    			# 		c=c+' '
				sent+=c
				i+=len(c)
			sent=sent.strip()
			# verify_sent=self.src_dict.string(src_token,bpe_symbol='@@').replace(' ','')
			# assert sent==verify_sent,'%s\n%s'%(sent,verify_sent)
			return sent,position_map

		device=src_tokens.device
		tgt_len=attn.size(1)
		bsz,src_len=src_tokens.size()
		eos=self.src_dict.eos()
		bos=self.src_dict.bos()
		assert attn.size(-1)==src_len
		
		mask=src_tokens.unsqueeze(1)
		mask=mask.eq(eos)|mask.eq(bos)
		if self.args.debugging:
    			print(step,'raw attn',attn)
		if not self.training:
			attn=torch.where(mask,torch.full_like(attn,0),attn)
			attn=attn/attn.sum(dim=-1,keepdim=True)
		num_idx=torch.argmax(attn,dim=-1).cpu()
		src_tokens=src_tokens.cpu()
		eon=self.num_dict.index(EON)
		pad=self.num_dict.pad()

		en_tensors=None
		position_maps,numsets=None,None
		src_num_map=None
		if incremental_state is not None:
			en_tensors = get_incremental_state(self, incremental_state, 'num_tensors')

			position_maps=get_incremental_state(self,incremental_state,'position_maps')
			numsets=get_incremental_state(self,incremental_state,'numsets')
			src_num_map=get_incremental_state(self,incremental_state,'src_num_map')
		if en_tensors is None:
			en_tensors = []
		if position_maps is None or numsets is None:
			src_sents=[]
			position_maps=[]
			for i in range(bsz):
				# print('src_tokens', self.src_dict.string(src_tokens[i],bpe_symbol='@@'))
				src_sent, position_map = build_src_sentence(src_tokens[i])
				# print('builded_tokens',src_sent)
				src_sents.append(src_sent)
				position_maps.append(position_map)
			self.timers[0].tic()
			numsets=findall_nums(src_sents)
			src_num_map=src_tokens.new(bsz,src_len,1).fill_(eon).long()
			for i in range(bsz):
				src_sent=src_tokens[i]
				last_str=''
				num_ts = torch.LongTensor([eon])
				for j in range(src_len):
					if src_sent[j]==0 or src_sent[j]==2:
						num_str=''
					else:
						pos = position_maps[i][j]
						num_str=self.find_num_str(pos,numsets[i])
					if num_str!=last_str:
						en=self.translate(num_str)
						if en:
							en=self.tokenize_num(en)
							num_ts=self.num_dict.encode_line(en,add_if_not_exist=False,append_eos=False).long()
							try:
								assert self.num_dict.unk() not in num_ts,'%s %s %s'%(self.src_dict.string(src_sent),en,num_ts)
							except AssertionError as e:
								print(e)
								num_ts=torch.LongTensor([eon])
						else:
							num_ts=torch.LongTensor([eon])
						last_str=num_str
					if src_num_map.size(-1)<num_ts.size(0):
						src_num_map=torch.cat([src_num_map,torch.zeros(bsz,src_len,num_ts.size(0)-src_num_map.size(-1)).fill_(eon).long()],dim=-1)
					src_num_map[i][j][:num_ts.size(0)]=num_ts
			src_num_map=src_num_map.to(device)
			# numsets[i].append((m.group(), m.start(), m.end()))
			# print(numsets)
			t=self.timers[0].toc()
			# if t>=10-1e-8:
			# 	print('timeout',src_sents,numsets)
			if incremental_state is not None:
				set_incremental_state(self,incremental_state,'position_maps',position_maps)
				set_incremental_state(self,incremental_state,'numsets',numsets)
				set_incremental_state(self,incremental_state, 'src_num_map',src_num_map)
		if self.training:
			src_num_map=src_num_map[:,:,0].unsqueeze(1).repeat(1,tgt_len,1)
			# print(src_num_map.size(),attn.size())
			probs=src_tokens.new(bsz,tgt_len,len(self.num_dict)).type_as(attn).fill_(0).to(device)
			# log_attn=torch.log(attn)
			# log_attn=torch.where(torch.isnan(log_attn) , torch.full_like(log_attn , -math.inf) , log_attn)
			probs=probs.scatter_add(dim=2,index=src_num_map,src=attn)
			probs=probs.log()
			assert not torch.isinf(probs).all()
			# print(probs.size())
			return probs
		# attn=attn.cpu()
		for i in range(bsz):
			num_sent=torch.LongTensor(0) if i>=len(en_tensors) else en_tensors[i]
			for j in range(tgt_len):
				if not self.training and step+j<len(num_sent):
					continue
				if self.args.debugging:
					print(step,'attn',attn[i,j])
				raw_pos=num_idx[i][j].item()
				num_tensor=src_num_map[i][raw_pos].to(num_sent)
				# self.timers[1].toc()
				self.timers[2].tic()
				num_sent=torch.cat([num_sent[:step+j],num_tensor[num_tensor!=eon],torch.LongTensor([eon]).to(num_sent)])
				if getattr(self.args,'debugging',False):
					print(step,'bsz',i,'pos',j,'result',num_tensor,self.num_dict.string(num_tensor),num_sent)
				self.timers[2].toc()
			num_sent=num_sent.long()
			# num_sent=num_sent.to(device)
			if i<len(en_tensors):
				en_tensors[i]=num_sent
			else:
				en_tensors.append(num_sent)
		# if DEBUGGING:
		# for i in range(3):
		# 	print('timer',i,":",self.timers[i].get())
		if incremental_state is not None:
			max_len=1
			set_incremental_state(self,incremental_state,'num_tensors',en_tensors)
		else:
			max_len=tgt_len
		assert en_tensors[0].dtype==torch.long
		padded_tensors=en_tensors[0].new(bsz,max_len).fill_(eon)
		# print(step,en_tensors)
		for i,t in enumerate(en_tensors):
			padded_tensors[i,:min(max_len,t.size(0)-step)].copy_(t[step:step+min(max_len,t.size(0)-step)])
		# print('padded_tensors',padded_tensors)
		padded_tensors=padded_tensors.to(device)
		one_hot=attn.new_zeros(bsz,max_len,len(self.num_dict)).fill_(-math.inf)
		padded_tensors=padded_tensors.view(bsz,max_len,1)
		#
		# if getattr(self.args,'debugging',False):
		# 	print(step,padded_tensors)
		one_hot=one_hot.scatter_(2,padded_tensors,0)
		if not requires_label:
			return one_hot
		label = padded_tensors.ne(eon).long()
		label=label[:,:tgt_len,:]-torch.cat([label.new_zeros(label.size(0),1,label.size(2)),label[:,:tgt_len-1,:]],dim=1)
		label.masked_fill_(label.lt(0),0)
		return one_hot,label

	def clean(self,incremental_state,step,gen_state):
		if incremental_state is None:
			return
		en_tensors=get_incremental_state(self,incremental_state,'num_tensors')
		if en_tensors is None:
			return
		# if DEBUGGING:
		# 	print(step,'clean',en_tensors,gen_state)
		for i,t in enumerate(en_tensors):
			if not gen_state[i]:
				t.fill_(self.num_dict.index(EON))
				en_tensors[i]=t[:step]
		# if DEBUGGING:
		# 	print(step,'after clean',en_tensors)
	def reorder_incremental_state(self,incremental_state,reorder_state):
		if incremental_state is None:
			return
		en_tensors=get_incremental_state(self,incremental_state,'num_tensors')
		position_maps=get_incremental_state(self,incremental_state,'position_maps')
		numsets=get_incremental_state(self,incremental_state,'numsets')
		src_num_map=get_incremental_state(self,incremental_state,'src_num_map')
		# if DEBUGGING:
		# 	print('reorder',en_tensors,reorder_state)
		new_en_tensors=[]
		new_position_maps=[]
		new_numsets=[]
		for idx in reorder_state:
			new_en_tensors.append(en_tensors[idx].clone())
			new_position_maps.append(position_maps[idx])
			new_numsets.append(numsets[idx])
		src_num_map=src_num_map.index_select(0,reorder_state)
		set_incremental_state(self,incremental_state,'num_tensors',new_en_tensors)
		set_incremental_state(self,incremental_state,'position_maps',new_position_maps)
		set_incremental_state(self,incremental_state,'numsets',new_numsets)
		set_incremental_state(self,incremental_state,'src_num_map',src_num_map)
		# if DEBUGGING:
		# 	print('after reorder',get_incremental_state(self,incremental_state,'num_tensors'))
	def find_num_str(self,pos,numset):
		for s,l,r in numset:
			if pos>=l and pos < r:
				return s
		return ''
	def tokenize_num(self,en):
		if not en:
			return ''
		# print(en)
		words = en.split()
		if len(words) > 1:
			return ' '.join([self.tokenize_num(w) for w in words])
		assert isinstance(words,list)
		word = words[0]
		if word in self.num_dict:
			# print('dict return',word)
			return word
		if word[-2:] in ['st','nd','rd','th']:
			res=' '.join(map(lambda x:x+'@@',word[:-2]))+' '+word[-2:]
		else:
			res=' '.join(map(lambda x:x+'@@',word))
			res=res[:-2]
		# print('return',res)
		return res
	def translate(self,cn):
		if cn=='' or cn is None:
			return ''

		num_type=get_chinese_num_type(cn)
		self.timers[1].tic()
		en=''
		if num_type=='time':
			en=time2en(cn)
		elif num_type=='date':
			en=date2en(cn)
		elif num_type=='ordinal':
			en=ordinal2en(cn)
		elif num_type=='percent':
			en=percent2en(cn)
		elif num_type=='age':
    			en=age2en(cn)
		elif num_type=='number':
			en=cn2en(cn)
		self.timers[1].toc()
		return en

if __name__=='__main__':
	rnt=RuleNumTranslator()
	print(rnt.translate('过往五年搜获的淫亵物品数量如下:一九八五45021一九八六12790一九八七6336一九八八7177一九八九55189总数126513'))
