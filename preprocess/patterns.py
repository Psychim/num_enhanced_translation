import re
from decimal import Decimal
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

digit_pattern=r'(?:[一二三四五六七八九零○]|\d)'
magni_pattern=r'(?: ?(?:(?<!两)十)| ?[百千万亿兆])'
num_sequence_pattern=r'(?:(?<!乱)(?:{},?)+(?!糟))'.format(digit_pattern) 
value_unit_pattern=r'(?:(?:{}|两){}+[零○]?)'.format(digit_pattern,magni_pattern)
chinese_value_pattern=r'(?:十?(?:{}+{}?)|(?:十{}?))'.format(value_unit_pattern,digit_pattern,digit_pattern)
integer_pattern=r'(?:{}|{}|两)'.format(chinese_value_pattern,num_sequence_pattern)
number_pattern=r'(?:(?!几){}(?:[点.]{}+)? ?多?{}*(?!几))'.format(
	integer_pattern,
	digit_pattern,
	magni_pattern,
)

year_pattern=r'(?:{} ?年代?)'.format(integer_pattern)
month_value_pattern=r'(?:{}|[十1][一二120]|十)'.format(digit_pattern)
month_pattern=r'(?:(?:{}|元) ?月份?)'.format(month_value_pattern)
day_value_pattern=r'(?:{}|[十1]{}?|(?:二十|2|廿){}|(?:三十|3)[01一]|二十|三十)'.format(digit_pattern,digit_pattern,digit_pattern)
day_pattern=r'(?:{} ?[日号])'.format(day_value_pattern)
date_pattern=r'(?:{}?{}?{}|{}?{}|{})'.format(
	year_pattern,month_pattern,day_pattern,
	year_pattern,month_pattern,
	year_pattern,
)
ordinal_pattern=r'(?:第 ?{})'.format(integer_pattern)
hour_pattern=r'(?:{} ?[点时])'.format(integer_pattern)
compiled_hour=re.compile(hour_pattern)
minute_pattern=r'(?:{} ?分|半)'.format(integer_pattern)
compiled_minute=re.compile(minute_pattern)
second_pattern=r'(?:{} ?秒)'.format(integer_pattern)
compiled_second=re.compile(second_pattern)
time_pattern=r'(?:{}?{}?{}|{}?{})'.format(
	hour_pattern,minute_pattern,second_pattern,
	hour_pattern,minute_pattern,
)
percent_pattern=r'(?:{} ?成)'.format(digit_pattern)
age_pattern=r'(?:{} ?旬)'.format(digit_pattern)
pattern=r'(?:{}|{}|{}|{}|{}|{})'.format(
	time_pattern,
	date_pattern,
	ordinal_pattern,
	percent_pattern,
	age_pattern,
	number_pattern
)
compiled_ordinal=re.compile(ordinal_pattern)
compiled_integer=re.compile(integer_pattern)
compiled_number=re.compile(number_pattern)
compiled_date=re.compile(date_pattern)
compiled_time=re.compile(time_pattern)
compiled_percent=re.compile(percent_pattern)
compiled_age=re.compile(age_pattern)
compiled_pattern=re.compile(pattern)

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
	if len(str(num/unit_val))-float_digit(num/unit_val)<=6:
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
		if v.group[0][0]=='元':
			return 1,v.group[0]			
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
def translate2arabic(cn):
	def extract(pattern):
		v=re.search(pattern,cn)
		if not v:
			return None
		v=v.group(0)
		if v=='半':
			return 30
		elif v[0]=='元':
			return 1
		v=compiled_integer.search(v).group(0)
		v=cn2an(v)
		return v
	num_type=get_chinese_num_type(cn)
	if num_type=='time':
		hour=extract(hour_pattern)
		minute=extract(minute_pattern)
		second=extract(second_pattern)
		val=[hour,minute,second]
		val=[v for v in val if v is not None]
	elif num_type=='date':
		year=extract(year_pattern)
		month=extract(month_pattern)
		day=extract(day_pattern)
		val=[year,month,day]
		val=[v for v in val if v is not None]	
	elif num_type=='ordinal':
		val=cn2an(compiled_integer.search(cn).group(0))
	elif num_type=='percent':
		val=cn2an(compiled_integer.search(cn).group(0))
		val*=10
	elif num_type=='age':
		val=cn2an(compiled_integer.search(cn).group(0))
		val*=10
	elif num_type=='number':
		val=cn2an(cn)
	else:
		val=None
	if isinstance(val,list) and len(val)==1:
			val=val[0]
	return val


en_scale_pattern=r'(?:thousand|million|billion|trillion)'
en_integer_pattern=r'(?:\d+(?:,?\d+)*)'
en_word_pattern=r'(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|one|two|three|four|five|six|seven|eight|nine|ten)'
en_real_pattern=r'(?:{}(?:\.\d+)?)'.format(en_integer_pattern)
en_num_pattern=r'(?:(?:{}|{})(?: -)?(?: {})?)'.format(en_real_pattern,en_word_pattern,en_scale_pattern)
en_ordinal_pattern=r'(?:\d*(?:1st|2nd|3rd)|\d+th)'
en_percent_pattern=r'(?:\d+ ?\%)'
en_day_pattern=r'(?:{}|(?:\d+))'.format(en_ordinal_pattern)
en_month_pattern=r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
en_date_pattern=r'(?:{} {} , {}|{} {}|{} {}|{})'.format(en_month_pattern,en_day_pattern,en_integer_pattern,en_month_pattern,en_day_pattern,en_day_pattern,en_month_pattern,en_month_pattern)
en_time_pattern=r'(?:{} : {} : {}|{} : {})'.format(en_integer_pattern,en_integer_pattern,en_integer_pattern,en_integer_pattern,en_integer_pattern)
en_decade_pattern=r'(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)'
en_pattern=r'(?:\b{}\b|\b{}\b|\b{}\b|\b{}\b|\b{}\b|\b{}\b)'.format(
	en_time_pattern,
	en_date_pattern,
	en_ordinal_pattern,
	en_percent_pattern,
	en_decade_pattern,
	en_num_pattern
)
compiled_en_pattern=re.compile(en_pattern)
def get_english_num_type(en):
	if re.match(en_time_pattern,en):
		return 'time'
	elif re.match(en_date_pattern,en):
		return 'date'
	elif re.match(en_ordinal_pattern,en):
		return 'ordinal'
	elif re.match(en_percent_pattern,en):
		return 'percent'
	elif re.match(en_decade_pattern,en):
		return 'decade'
	elif re.match(en_num_pattern,en):
		return 'number'
	return ''
en_month_table={'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,'jan':1,
				'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
en_decade_table={'twenties':20,'thirties':30,'forties':40,'fifties':50,'sixties':60,'seventies':70,'eighties':80,'nineties':90}
en_scale_table={'thousand':1000,'million':1000000,'billion':1000000000,'trillion':1000000000000}
en_word_table={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,
				'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20}
def en_translate2arabic(en):
	num_type=get_english_num_type(en)
	if num_type=='time':
		vals=en.split(':')
		val=list(map(Decimal,vals))
	elif num_type=='date':
		month=re.search(en_month_pattern,en)
		if month:
			month=en_month_table[month.group(0)]
		vals=re.findall(en_num_pattern,en)
		if len(vals)==2:
			day,year=list(map(Decimal,vals))
		elif len(vals)==1:
			v=Decimal(vals[0])
			if v<=31:
				day=v 
				year=None 
			else:
				year=v 
				day=None 
		else:
			day,year=None,None
		val=[year,month,day]
		val=[v for v in val if v is not None]
		
	elif num_type=='decade':
		val=Decimal(en_decade_table[en])
	elif num_type=='ordinal' or num_type=='percent' or num_type=='number':
		
		scale=re.search(en_scale_pattern,en)
		if scale:
			scale=scale.group(0)
			en=en.strip()[:-len(scale)]
			scale=en_scale_table[scale]
		en=en.replace(',','').replace('-','')
		val=re.search(en_num_pattern,en).group(0)
		if val in en_word_table:
			val=Decimal(en_word_table[val])
		else:
			val=Decimal(val)
		if scale:
			val*=scale
	else:
		val=None 
	if isinstance(val,list) and len(val)==1:
			val=val[0]	
	return val
