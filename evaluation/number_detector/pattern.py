import sys
sys.path.append('/home/data_ti4_c/lijh/projects/numeral_translation/scripts')
import re
import argparse
from collections import defaultdict
import pdb
import numpy
from utils import Digit, Date, Ordinal


word_boundary = r'\b'
textual_digit = r'(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|hundred|thousand|million|billion|trillion|one|two|three|four|five|six|seven|eight|nine|ten)'
simple_numbers = '{}{}(?:-|{})'.format(word_boundary,textual_digit,word_boundary)
complex_numbers = '(?:{})(?:-{}{}| - {}{}| and {}{}|{}{}{}| {}{}{})+'.format(
    textual_digit,
    textual_digit,word_boundary,
    textual_digit,word_boundary,
    textual_digit,word_boundary,
    word_boundary,textual_digit,word_boundary,
    word_boundary,textual_digit,word_boundary)
order_number = r'\d{1,2}(?:st|nd|rd|th)'
order_number_or_number = '(?:{}|{})'.format(order_number,r'\d{1,4}')
month_pattern = r'(?:\bjanuary\b|\bfebruary\b|\bmarch\b|\bapril\b|\bjune\b|\bjuly\b|\baugust\b|\bseptember\b|\boctober\b|\bnovember\b|\bdecember\b|\bjan\b|\bfeb\b|\bmar\b|\bapr\b|\bjun\b|\bjul\b|\baug\b|\bsep\b|\boct\b|\bnov\b|\bdec\b)'
year_pattern = r'\d{4}'
decade_pattern = r'(?:\b(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties|(?:\d{2,4} ?\'?s))\b)'
extract_decade_pattern=r'(\d{2,4}) ?\'?s'
date_pattern = '(?:(?:{} {}|{} {}|{}|{}|{})(?: , {}| {})?)'.format(
    month_pattern,order_number_or_number,
    order_number_or_number,month_pattern,
    month_pattern,
    decade_pattern,
    order_number,
    year_pattern,
    year_pattern,
)

# unit_pattern = r'(?:months?\b|tons?\b|kilograms?\b|million\b|billion\b|trillion\b|hours?\b|seconds?\b|kilometers?\b|%)'
# unit_pattern = r'(?:million\b|billion\b|trillion\b|%)'
# digit_pattern = r'(?:\$ )?\d+(?:,\d+)?\.?\d*'
digit_pattern = r'\d+(?:,\d+| \d+)*\.?\d*'

time_pattern = r'\d{1,2} : \d{2}(?: : \d{2})?'#(?: p\.m\.|a\.m\.)?'

#pattern=complex_numbers
pattern = '(?:{}|{}|{} (?:- )?{}|{}|{}|{})'.format(
    time_pattern,
    decade_pattern,
    digit_pattern,simple_numbers,
    complex_numbers,
    simple_numbers,
#    order_number,
    digit_pattern,
    # unit_pattern
)

pattern_types = {
    'time': time_pattern,
#    'date': date_pattern,
    'decade': decade_pattern,
    'digit_textual_number': '(?:{} (?:- )?{})'.format(digit_pattern, simple_numbers),
    'complex_textual_number': complex_numbers,
    'simple_textual_number': simple_numbers,
    'digit': digit_pattern,
}

decade_table ={
    'twenties':1920,
    'thirties':1930,
    'forties':1940,
    'fifties':1950,
    'sixties':1960,
    'seventies':1970,
    'eighties':1980,
    'nineties':1990,
}

month_table = {
    'jan':1,
    'january':1,
    'feb':2,
    'february':2,
    'mar':3,
    'march':3,
    'apr':4,
    'april':4,
    'jun':6,
    'june':6,
    'jul':7,
    'july':7,
    'aug':8,
    'august':8,
    'sep':9,
    'september':9,
    'oct':10,
    'october':10,
    'nov':11,
    'november':11,
    'dec':12,
    'december':12}

number_to_value_table = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'thousand': 1000,
    'million': 1000000
}

scale_table = {
    'hundred': 100,
    'thousand': 1000,
    'million': 1000000,
    'billion': 1000000000,
    'trillion': 1e12
}


en_prog = re.compile(pattern)


def get_number_type(a):
    for t, t_pattern in pattern_types.items():
        try:
            re.findall(t_pattern,a)
        except:
            pdb.set_trace()
        if len(re.findall(t_pattern,a)) != 0:
            return t
    return None


def standarlize_date(t):
    if t in decade_table:
        return Digit(decade_table[t])
    month = re.findall(month_pattern,t)
    day = re.findall(r'\b\d{1,2}(?:st|nd|rd|th)?\b',t)
    year = re.findall(year_pattern,t)

    if len(day) == 0:
        day = None
    else:
        day = day[0]

    if len(month) == 0:
        month = None
    else:
        month = month[0]

    if len(year) == 0:
        year = None
    else:
        year = year[0]

    if day is not None and (day.endswith('th') or day.endswith('nd') or day.endswith('rd') or day.endswith('st')):
        day = day[:-2]
    if month is not None and month in month_table:
        month = month_table[month]
    if year is not None:
        year = int(year)
    if month is not None:
        month = int(month)
    if day is not None:
        day = int(day)
    return Date(year,month,day)


def standarlize_time(t):
    return t


def standarlize_digit(t):
    # remove comma
    t = t.replace(',','').replace(' , ','')
    t = t.replace('s','')
    try:
        ret = float(t)
        return Digit(ret)
    except:
        return None


def standarlize_digit_textual(t):
    splited = t.split()
    if len(splited)==3 and splited[1]=='-':
        splited=[splited[0],splited[2]]
    if len(splited) != 2:
        return None
    digit, textual = splited[0], splited[1]
    digit_value = standarlize_digit(digit).d
    try:
        scale = scale_table[textual]
        return Digit(digit_value * scale)
    except:
        return None


def standarlize_complex_textual(t):
    value = 0
    prev_type = None
    for x in t.split():
        if x[-1]=='-':
            x=x[:-1]
        if '-' in x:
            splited = x.split('-')
            if splited[1] in scale_table:
                value += number_to_value_table[splited[0]] * scale_table[splited[1]]
            else:
                value += number_to_value_table[splited[0]] + number_to_value_table[splited[1]]
            prev_type = 'digit'
        elif x in scale_table:
            value *= scale_table[x]
        elif x in number_to_value_table:
            if prev_type == 'digit':
                return None
            else:
                value += number_to_value_table[x]
                prev_type = 'digit'
        elif x == 'and':
            if prev_type == 'digit':
                return None
            else:
                continue
        else:
            return None
    return Digit(value)


def standarlize_simple_textual(t):
    if t[-1]=='-':
        t=t[:-1]
    if t in ['first','second','third','fourth','fifth','sixth','seventh','eighth','nineth','tenth']:
        return Ordinal(number_to_value_table[t])
    elif t in ['hundred','thousand','million','billion','trillion']:
        return None
    else:
        return Digit(number_to_value_table[t])


def standarlize(t):
    error_type = get_number_type(t)
    if error_type == 'time':
        ts = standarlize_time(t)
    elif error_type == 'decade':
        if t in decade_table:
            ts = Digit(float(decade_table[t]))
        else:
            m=re.match(extract_decade_pattern,t)
            if m and len(m.group())>=2:
                v=int(m.group(1))
                if v<100:
                     ts = Digit(1900+v)
                else:
                     ts = Digit(v)
            else:
                return None
    elif error_type == 'date':
        ts = standarlize_date(t)
    elif error_type == 'digit_textual_number':
        ts = standarlize_digit_textual(t)
    elif error_type == 'complex_textual_number':
        try:
            ts = standarlize_complex_textual(t)
        except:
            return None
    elif error_type == 'simple_textual_number':
        ts = standarlize_simple_textual(t)
    elif error_type == 'digit':
        ts = standarlize_digit(t)
    else:
        return None
    return ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    args = parser.parse_args()

    print(en_prog.findall('one million'))
    exit()
    # print(re.findall(complex_numbers,' and twenty'))
    types_dict = defaultdict(lambda: [])
    with open(args.input_path) as f_input:
        for line in f_input:
            hyp_list = en_prog.findall(line.rstrip())
            for n in hyp_list:
                types_dict[get_number_type(n)].append(n)
                print(standarlize(n))

    # for t in types_dict['complex_textual_number']:
        # print(t, standarlize_complex_textual(t))

    # for t in types_dict['simple_textual_number']:
    #     print(t, standarlize_simple_textual(t))
