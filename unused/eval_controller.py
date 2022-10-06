import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-i',type=str)
parser.add_argument('-r',type=str)
args=parser.parse_args()
with open(args.r) as fr , \
    open(args.i) as fi:
    tp,fp,tn,fn=[0]*4
    for li,lr in zip(fi.readlines(),fr.readlines()):
        vi=li.strip().split()
        vr=lr.strip().split()
        assert len(vi)==len(vr)
        tp+=sum(map(lambda x:x[0]=='1' and x[1]=='1',zip(vi,vr)))
        fp+=sum(map(lambda x:x[0]=='1' and x[1]=='0',zip(vi,vr)))
        tn+=sum(map(lambda x:x[0]=='0' and x[1]=='0',zip(vi,vr)))
        fn+=sum(map(lambda x:x[0]=='0' and x[1]=='1',zip(vi,vr)))
    prec=tp/(tp+fp)
    rec=tp/(tp+fn)
    f1=2*prec*rec/(prec+rec)
    print('prec',prec,'rec',rec,'f1',f1) 