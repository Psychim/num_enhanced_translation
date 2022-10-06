import re
import argparse
import pdb
import copy
from collections import defaultdict
from pattern import (
    pattern,
    get_number_type,
    standarlize
)

from utils import Date,Digit,Ordinal

month_table=[]
#month_table = [
#        'january','jan',
#        'february','feb',
#        'march','mar',
#        'april','apr',
#        'june','jun',
#        'july','jul',
#        'august','aug',
#        'september','sep',
#        'october','oct',
#        'november','nov',
#        'december','dec']


prog = re.compile(pattern)


class F1Score:
    def __init__(self,correct=0,pred=0,gold=0):
        self.correct = correct
        self.pred = pred
        self.gold = gold

    def __add__(self,other):
        self.correct += other.correct
        self.pred += other.pred
        self.gold += other.gold

        return self

    def __repr__(self):
        p = self.correct / self.pred
        r = self.correct / self.gold
        f = 2 * p * r / (p+r)
        return 'P={:.2f}, R={:.2f}, F={:.2f}'.format(p*100,r*100,f*100)


def evaluate_EM(a,b,i):
    correct = 0
    la,lb = copy.deepcopy(a),copy.deepcopy(b)
    gold = len(la)
    pred = len(lb)
    for x in la:
        if x in lb:
            correct += 1
            lb.remove(x)
    return correct, pred, gold


def evaluate_value(a,b,i):
    correct = 0
    la,lb = [],[]
    for x in a:
        la.append(standarlize(x))
        if i==763:
            print(standarlize(x),type(standarlize(x)))
    for x in b:
        lb.append(standarlize(x))
        if i==763:
           print(standarlize(x),type(standarlize(x)))
    gold,pred = len(la),len(lb)
    for x in la:
        if x in lb:
            correct += 1
            lb.remove(x)
    return correct, pred, gold


def filter_list(xs):
    new_xs = []
    for x in xs:
        st_x = standarlize(x)
        if st_x is None or isinstance(st_x, Ordinal):
            continue
        try:
            if float(st_x) <= 10 and float(st_x) >= 0:
                continue
        except:
            pass
        if x == 'one' or x == 'first' or x == 'two' or x == '1' or x == '2' or x == 'second':
            continue
        else:
            new_xs.append(x)
    return new_xs


def main(args):
    if 'evaluate' in args.mode:
        evaluate_fn = evaluate_EM if args.evaluate_mode == 'EM' else evaluate_value
        fout = open(args.output_path,'w') if args.output_path is not None else None
        f1_score = F1Score()
        with open(args.input_path) as f_input, open(args.reference_path) as f_reference:
            for i,(line_a,line_b) in enumerate(zip(f_reference,f_input)):
                ref_list = prog.findall(line_a.rstrip())
                hyp_list = prog.findall(line_b.rstrip())
                if i==763:
                    print(line_b)
                    print(ref_list,hyp_list)
                ref_list = filter_list(ref_list)
                hyp_list = filter_list(hyp_list)
                if i==763:
                    print(ref_list,hyp_list)
                try:
                    correct, pred, gold = evaluate_fn(copy.deepcopy(ref_list),copy.deepcopy(hyp_list),i)
                except:
                    pdb.set_trace()
                f1_score += F1Score(correct,pred,gold)
                if len(ref_list) == 0 and len(hyp_list) == 0:
                    continue
                if fout is not None and (correct != pred or correct != gold):
                # if fout is not None:
                    fout.write(str(i) + '\n')
                    fout.write(line_a)
                    fout.write(line_b)
                    if len(ref_list) == 0:
                        fout.write('None\n')
                    else:
                        fout.write('\t'.join(ref_list) + '\n')
                    if len(hyp_list) == 0:
                        fout.write('None\n')
                    else:
                        fout.write('\t'.join(hyp_list) + '\n')

                    standarlized_ref_list = []
                    for x in ref_list:
                        standarlized_ref_list.append(standarlize(x))
                    if len(standarlized_ref_list) == 0:
                        fout.write('None\n')
                    else:
                        fout.write('\t'.join(map(str,standarlized_ref_list)) + '\n')

                    standarlized_hyp_list = []
                    for x in hyp_list:
                        standarlized_hyp_list.append(standarlize(x))
                    if len(standarlized_hyp_list) == 0:
                        fout.write('None\n')
                    else:
                        fout.write('\t'.join(map(str,standarlized_hyp_list)) + '\n')
                    fout.write(str(evaluate_EM(ref_list,hyp_list,i)) + '\n')
                    fout.write(str(evaluate_value(ref_list,hyp_list,i)) + '\n')
                    fout.write('\n')
        if args.evaluate_mode == 'EM':
            print('EM: {}'.format(f1_score))
        elif args.evaluate_mode == 'value':
            print('Value :{}'.format(f1_score))

    elif args.mode == 'parse':
        cnt = 0
        total_words = 0
        types_dict = defaultdict(lambda: [])
        with open(args.input_path) as f_input:
            if args.output_path is not None:
                fout = open(args.output_path,'w')
            else:
                fout = None
            for line in f_input:
                hyp_list = prog.findall(line.rstrip())
                for n in hyp_list:
                    types_dict[get_number_type(n)].append(n)
                cnt += len(hyp_list)
                total_words += len(line.split())
                if len(hyp_list) == 0 and fout is not None:
                    fout.write('None\n')
                elif fout is not None:
                    fout.write('\t'.join(hyp_list) + '\n')
        print(cnt)
        print(total_words)
        print(cnt / total_words)
        if fout is not None:
            fout.close()

        simple_number_count = len(types_dict['simple_textual_number'])
        date_count = len(types_dict['date'])
        complex_number_count = len(types_dict['complex_textual_number']) + len(types_dict['digit_textual_number'])

        for n in types_dict['digit']:
            try:
                value = standarlize(n).d
            except:
                continue
            if int(value) == float(value) and value < 100:
                simple_number_count += 1
            else:
                complex_number_count += 1

        print('Simple: {:.4f}, {:.4f}'.format(simple_number_count, simple_number_count / cnt))
        print('Date: {:.4f}, {:.4f}'.format(date_count, date_count/cnt))
        print('Complex: {:.4f}, {:.4f}'.format(complex_number_count, complex_number_count/cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--reference_path')
    parser.add_argument('--output_path')
    parser.add_argument('--mode')
    parser.add_argument('--evaluate-mode')

    args = parser.parse_args()
    main(args)
