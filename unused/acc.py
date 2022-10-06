import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-f',type=str,default="result.txt")
parser.add_argument('-r',type=int)
args=parser.parse_args()
f=open(args.f,encoding='utf-8')
tgt={}
hyp={}
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
tp,tn,fp,fn=[0]*4
right_num=0
print(len(hyp))
relax=args.r
for idx,h in hyp.items():
	t=tgt[idx]
	h=h.split()
	t=t.split()
	h.extend(['0']*(len(t)-len(h)))
#	print(h,t)
	assert len(h)==len(t),'%d %d %d %s %s'%(idx,len(h),len(t),h,t)
	
	l=len(h)
	tp+=sum([h[i]=='1' and ',' in t[i:i+relax]  for i in range(l)])
	tn+=sum([h[i]=='0' and t[i]=='the' for i in range(l)])
	fp+=sum([h[i]=='1' and not ',' in t[i:i+relax] for i in range(l)])
	fn+=sum([h[i]=='0' and t[i]!='the' for i in range(l)])
print(tp,tn,fp,fn)
recall=tp/(tp+fn)
prec=tp/(tp+fp)
acc=(tp+tn)/(tp+tn+fp+fn)
print('Recall: %.4f Precision: %.4f Accuracy: %.4f f1: %.4f'%(recall,prec,acc,2*prec*recall/(recall+prec)))
print('%.4f'%(right_num/len(hyp)))
