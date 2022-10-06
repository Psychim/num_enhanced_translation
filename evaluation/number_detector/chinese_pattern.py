import argparse
import re
import pdb
from collections import defaultdict
import chinese2digits as c2d
from number_detector.utils import Date, Digit, Ordinal
from tqdm import tqdm


def c2d_fn(c):
    return int(c2d.takeNumberFromString(c)['digitsStringList'][0])

scale_to_number = {
    '百万': 1000000,
    '千万': 10000000,
    '百亿': 10000000000,
    '千亿': 100000000000,
    '万亿': 1000000000000,
    '十': 10,
    '百': 100,
    '千': 1000,
    '万': 10000,
    '亿': 100000000,
    '兆': 10000000000000,
}


simple_char=r'(?:一|二|三|四|五|六|七|八|九|十|两)'
simple_scale = r'(?:百万|千万|百亿|千亿|万亿|十|百|千|万|亿|兆)'

day_digit = r'\d{1,2}'
day = '(?:{}|十{}*|二十{}*|三十|三十一|{})日'.format(simple_char,simple_char,simple_char,day_digit)
month = r'(?:元|一|二|三|四|五|六|七|八|九|十|十一|十二|1|2|3|4|5|6|7|8|9|10|11|12)月'
year = r'(?:[一二三四五六七八九0零\u25cb]{3,4}|\d{3,4})年'

date = '(?:{}{}{}|{}{}|{}|{}|{})'.format(
    year,month,day,
    month,day,
    year,
    month,
    day
)

digit = r'\d+(?:,\d+|\s\d+)?\.?\d*'

textual = '(?:(?:{}(?:{}|{}|零{}{}|零{})*)(?:点{}+)?|零点{}+)'.format(simple_char,simple_char,simple_scale,simple_char,simple_scale,simple_char,simple_char,simple_char)

digit_textual = '(?:{}{})+'.format(digit,simple_scale)

ordinal = '第(?:{}|{})'.format(textual,digit)

pattern = '(?:{}|{}|{}|{}|{})+'.format(
    digit_textual,
    ordinal,
    date,
    textual,
    digit)


pattern_types = {
    'digit_textual': digit_textual,
    'ordinal': ordinal,
    'date': date,
    'textual': textual,
    'digit': digit
}

zh_prog = re.compile(pattern)


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
    if '两' in t:
        return None
    mm = re.findall(month,t)
    dd = re.findall(day,t)
    yy = re.findall(year,t)

    if len(dd) == 0:
        dd = None
    else:
        try:
            dd = c2d_fn(dd[0][:-1])
        except:
            pdb.set_trace()

    if len(mm) == 0:
        mm = None
    else:
        if mm[0] == '元月':
            mm = 1
        else:
            mm = c2d_fn(mm[0][:-1])

    if len(yy) == 0:
        yy = None
    else:
        try:
            yy = int(yy[0][:-1])
        except:
            yy = yy[0][:-1].replace('\u25cb','零').replace('0','零')
            yy = c2d_fn(yy)

    return Date(yy,mm,dd)


def standarlize_digit(t):
    t = t.replace(',','').replace(' , ','')
    try:
        ret = float(t)
        return Digit(ret)
    except:
        return None


def standarlize_digit_textual(t):
    digits = re.findall(digit,t)
    scales = re.findall(simple_scale,t)
    try:
        ret = Digit(float(c2d.takeNumberFromString(digits[0])['digitsStringList'][0]) * scale_to_number[scales[0]])
        return ret
    except:
        pdb.set_trace()


def standarlize_ordinal(t):
    try:
        ret = Ordinal(c2d_fn(t[1:]))
        return ret
    except:
        return None


def standarlize_textual(t):
    t = t.replace('两','二')
    if '兆' in t:
        splited = t.split('兆')
        if splited[-1] != '':
            value = int(c2d.takeNumberFromString(splited[0])['digitsStringList'][0]) * scale_to_number['兆'] + int(c2d.takeNumberFromString(splited[1])['digitsStringList'][0])
        else:
            try:
                value = int(c2d.takeNumberFromString(splited[0])['digitsStringList'][0]) * scale_to_number['兆']
            except:
                pdb.set_trace()
    else:
        try:
            value = float(c2d.takeNumberFromString(t)['digitsStringList'][0])
            return Digit(value)
        except:
            return None


def standarlize(t):
    error_type = get_number_type(t)
    if error_type == 'digit_textual':
        ts = standarlize_digit_textual(t)
    elif error_type == 'ordinal':
        ts = standarlize_ordinal(t)
    elif error_type == 'date':
        ts = standarlize_date(t)
    elif error_type == 'textual':
        ts = standarlize_textual(t)
    elif error_type == 'digit':
        ts = standarlize_digit(t)
    else:
        return None
    return ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')

    # print(zh_prog.findall('一千零三十五万'))
    print(standarlize('十三点九'))
    print(standarlize('2005年3月24日'.replace(' ','')))
    exit()
    args = parser.parse_args()

    types_dict = defaultdict(lambda: [])
    cnt = 0
    with open(args.input_path) as f, open(args.output_path,'w') as fout:
        for line in tqdm(f):
            nums = zh_prog.findall(line.rstrip().replace(' ',''))
            for n in nums:
                if "一月" in n:
                    cnt += 1
                types_dict[get_number_type(n)].append(n)
                # print(standarlize(n))
            if len(nums) == 0:
                fout.write('None\n')
            else:
                fout.write('\t'.join(nums) + '\n')
    print(cnt)

    # for t in types_dict['textual']:
    #     print(t,standarlize_textual(t))
