from chinese_pattern import standarlize as standarlize_zh


class NumTranslator:
    def __init__(self):
        pass

    @staticmethod
    def translate(zh_num):
        n = standarlize_zh(zh_num)
        return n.express('en')


if __name__ == '__main__':
    translator = NumTranslator()
    while True:
        a = input()
        print(translator.translate(a))
