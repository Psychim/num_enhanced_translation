import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-f',type=str,default="result.txt")
parser.add_argument('-r',type=int)
args=parser.parse_args()
f=open(args.f,encoding='utf-8')
tgt={}
hyp={}
sco={}
for line in f.readlines():
	line=line.strip()
	if line.startswith('T-'):
		idx,sents=line.split('\t')
		idx=int(idx[2:])
		tgt[idx]=sents
	elif line.startswith('H-'):
		idx,score,sents=line.split('\t')
		idx=int(idx[2:])
		hyp[idx]=sents
	elif line.startswith('P-'):
		idx,sents=line.split('\t')
		idx=int(idx[2:])
		sco[idx]=sents
tp,e1,e2=0,0,0
relax=args.r
for idx,h in hyp.items():
	t=tgt[idx]
	p=sco[idx]
	h=h.split()
	t=t.split()
	p=p.split()
	t=int(t[0])
	p=sum(map(float,p[:-1]))
	if int(p+0.5)==t:
		tp+=1
	else:
		print(idx)
	e1+=abs(int(p+0.5)-t)
	e2+=abs(p-t)
total=len(hyp)
print(tp/total,e1/total,e2/total)
