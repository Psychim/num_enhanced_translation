import argparse
import random

num_dict = { 0: "零", 1: "一", 2: "二", 3: "三", 4: "四",
             5: "五", 6: "六", 7: "七", 8: "八", 9: "九" }
unit_map = [ ["", "十", "百", "千"],       ["万", "十万", "百万", "千万"],
             ["亿", "十亿", "百亿", "千亿"], ["兆", "十兆", "百兆", "千兆"] ]
unit_step = ["万", "亿", "兆"]


en_ordinal_expression = {
    1: 'first',
    2: 'second',
    3: 'third',
    4: 'fourth',
    5: 'fifth',
    6: 'sixth',
    7: 'seventh',
    8: 'eighth',
    9: 'ninth',
    10: 'tenth',
}

en_month_expression = {
    1: 'january',
    2: 'february',
    3: 'march',
    4: 'april',
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'august',
    9: 'september',
    10: 'october',
    11: 'november',
    12: 'december'
}

zh_num_expression = {
    '1':'一',
    '2':'二',
    '3':'三',
    '4':'四',
    '5':'五',
    '6':'六',
    '7':'七',
    '8':'八',
    '9':'九',
    '10':'十',
    '11':'十一',
    '12':'十二',
    '0':'零'
}


class number_to_chinese():
    """
       codes reference: https://github.com/tyong920/a2c
    """
    def __init__(self):
        self.result = ""

    def number_to_str_10000(self, data_str):
        """一万以内的数转成大写"""
        res = []
        count = 0
        # 倒转
        str_rev = reversed(data_str)  # seq -- 要转换的序列，可以是 tuple, string, list 或 range。返回一个反转的迭代器。
        for i in str_rev:
            if i is not "0":
                count_cos = count // 4  # 行
                count_col = count % 4   # 列
                res.append(unit_map[count_cos][count_col])
                res.append(num_dict[int(i)])
                count += 1
            else:
                count += 1
                if not res:
                    res.append("零")
                elif res[-1] is not "零":
                    res.append("零")
        # 再次倒序，这次变为正序了
        res.reverse()
        # 去掉"一十零"这样整数的“零”
        if res[-1] is "零" and len(res) is not 1:
            res.pop()

        return "".join(res)

    def number_to_str(self, data):
        """分段转化"""
        assert type(data) == float or int
        data_str = str(data)
        len_data = len(str(data_str))
        count_cos = len_data // 4  # 行
        count_col = len_data-count_cos*4  # 列
        if count_col > 0: count_cos += 1

        res = ""
        for i in range(count_cos):
            if i==0:
                data_in = data_str[-4:]
            elif i==count_cos-1 and count_col>0:
                data_in = data_str[:count_col]
            else:
                data_in = data_str[-(i+1)*4:-(i*4)]
            res_ = self.number_to_str_10000(data_in)
            res = res_ + unit_map[i][0] + res
        return res

    def decimal_chinese(self, data):
        assert type(data) == float or int
        data_str = str(data)
        if "." not in data_str:
            res = self.number_to_str(data_str)
        else:
            data_str_split = data_str.split(".")
            if len(data_str_split) is 2:
                res_start = self.number_to_str(data_str_split[0])
                res_end = "".join([num_dict[int(number)] for number in data_str_split[1]])
                res = res_start + random.sample(["点"], 1)[0] + res_end
            else:
                res = str(data)
        if res.startswith('一十'):
            res = res[1:]
        return res


class Date:
    def __init__(self,yy,mm,dd):
        self.yy = yy
        self.mm = mm
        self.dd = dd
        self.ntc = number_to_chinese()

    def __eq__(self,other):
        if type(other) != Date:
            return False

        return (self.yy == other.yy) and (self.mm == other.mm) and (self.dd == other.dd)

    def __repr__(self):
        return '{}-{}-{}'.format(self.yy,self.mm,self.dd)

    def express(self,lang):
        if lang == 'en':
            string = ''
            if self.mm is not None:
                string += en_month_expression[self.mm] + ' '
            if self.dd is not None:
                string += str(self.dd) + ' '
            if self.yy is not None:
                if string == '':
                    string += str(self.yy)
                else:
                    string += ', ' + str(self.yy)
            return string.strip()
        elif lang == 'zh':
            string = ''
            if self.yy is not None:
                for i in str(self.yy):
                    string += zh_num_expression[i]
                string += ('' if random.random() < 0.5 else ' ') + '年 '
            if self.mm is not None:
                string += zh_num_expression[str(self.mm)] + '月 '
            if self.dd is not None:
                string += self.ntc.number_to_str(self.dd) + '日'
            return string.strip().replace('一十','十')


class Digit:
    def __init__(self,d):
        self.d = d

    def __eq__(self,other):
        if type(other) != Digit:
            return False
        return self.d == other.d

    def __repr__(self):
        return str(self.d)

    def express(self,lang):
        if lang == 'en':
            return self.d
    def __float__(self):
        return float(self.d)
    def __int__(self):
        return int(self.d)

class Ordinal:
    def __init__(self,d):
        self.d = d

    def __eq__(self,other):
        if type(other) != Ordinal:
            return False
        else:
            return self.d == other.d

    def __repr__(self):
        return 'Ordinal: {}'.format(self.d)

    def express(self,lang):
        if lang == 'en':
            if self.d in en_ordinal_expression:
                return en_ordinal_expression[self.d]
            else:
                if self.d % 10 == 1:
                    return str(self.d) + 'st'
                elif self.d % 10 == 2:
                    return str(self.d) + 'nd'
                elif self.d % 10 == 3:
                    return str(self.d) + 'rd'
                else:
                    return str(self.d) + 'th'
