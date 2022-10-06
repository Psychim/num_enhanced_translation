import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-i',type=str)
parser.add_argument('-o',type=str)
args=parser.parse_args()
path_c=args.i
path_o=args.o
hs={}
with open(path_c,'r',encoding='utf-8') as f:
	for l in f.readlines():
		l=l.strip()
		if l.startswith('H-'):
			idx,score,hyp=l[2:].split('\t')
			idx=int(idx)
			hs[idx]=hyp.replace('@@ ','')
with open(path_o,'w',encoding='utf-8') as fout:
	for i in range(len(hs.keys())):
		fout.write(hs[i]+'\n')

