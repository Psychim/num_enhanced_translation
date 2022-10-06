import re
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

year_pattern=r'(?:{} ?年)'.format(integer_pattern)
month_value_pattern=r'(?:{}|[十1][一二120]|十)'.format(digit_pattern)
month_pattern=r'(?:{} ?月份?)'.format(month_value_pattern)
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
if __name__=='__main__':
    	print(re.findall(pattern,'五亿三千两百万'))