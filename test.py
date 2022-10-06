import re
digit_pattern=r'[一二三四五六七八九零两○\d]'
magni_pattern=r'[十百千万亿兆]'
integer_pattern=r'(?:(?:{}|{})+)'.format(digit_pattern,magni_pattern)
number_pattern=r'(?:{}(?:[点.]{}+)?{}?)'.format(
	integer_pattern,
	digit_pattern,
	magni_pattern,
)
year_pattern=r'(?:{}年)'.format(integer_pattern)
month_value_pattern=r'(?:{}|[十1][一二12]|十)'.format(digit_pattern)
month_pattern=r'(?:{}月份?)'.format(month_value_pattern)
day_value_pattern=r'(?:{}|[十1]{}?|(?:二十|2|廿){}|(?:三十|3)[01一]|二十|三十)'.format(digit_pattern,digit_pattern,digit_pattern)
day_pattern=r'(?:{}[日号])'.format(day_value_pattern)
date_pattern=r'(?:{}|{}?{}|{}?{}?{})'.format(
	year_pattern,
	year_pattern,month_pattern,
	year_pattern,month_pattern,day_pattern
)
ordinal_pattern=r'(?:第{})'.format(integer_pattern)
hour_pattern=r'(?:{}[点时])'.format(integer_pattern)
compiled_hour=re.compile(hour_pattern)
minute_pattern=r'(?:{}分)'.format(integer_pattern)
compiled_minute=re.compile(minute_pattern)
second_pattern=r'(?:{}秒)'.format(integer_pattern)
compiled_second=re.compile(second_pattern)
time_pattern=r'(?:{}?{}|{}?{}?{})'.format(
	hour_pattern,minute_pattern,
	hour_pattern,minute_pattern,second_pattern
)
percent_pattern=r'(?:{}成)'.format(number_pattern)
pattern=r'(?:{}|{}|{}|{}|{})'.format(
	time_pattern,
	date_pattern,
	ordinal_pattern,
	percent_pattern,
	number_pattern
)
compiled_ordinal=re.compile(ordinal_pattern)
compiled_integer=re.compile(integer_pattern)
compiled_number=re.compile(number_pattern)
compiled_date=re.compile(date_pattern)
compiled_time=re.compile(time_pattern)
compiled_percent=re.compile(percent_pattern)
compiled_pattern=re.compile(pattern)

s='过往五年搜获的淫亵物品数量如下:一九八五45021一九八六12790一九八七6336一九八八7177一九八九55189总数126513'

m=compiled_date.findall(s)

print (m)