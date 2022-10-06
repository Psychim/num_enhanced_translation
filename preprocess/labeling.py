import re 
from patterns import *
def detokenize_sent(tokens):
    sent=''
    position_map=[]
    i=0
    for c in tokens:
        position_map.append(i)
        if c.endswith('@@'):
            c=c[:-2]
        else:
        		c=c+' '
        sent+=c
        i+=len(c)
    sent=sent.strip()
    # verify_sent=self.src_dict.string(src_token,bpe_symbol='@@').replace(' ','')
    # assert sent==verify_sent,'%s\n%s'%(sent,verify_sent)
    return sent,position_map

def findall_nums(sent):
	numsets = []
	# print(joined_sents)
	matches = compiled_en_pattern.finditer(sent)
	# print(src_sents[i])
	for m in matches:
		pos = m.start()
		if m.group()=='may':
			continue
		numsets.append((m.group(), m.start(), m.end()))
	return numsets

with open('mt04.bpe.en') as fen:
    with open('mt04.bpe.label','w') as ofl:
        for le in fen.readlines():
            le=le.strip()
            tokens=le.split()
            sent,pmap=detokenize_sent(tokens)
            numset=findall_nums(sent)
            print(numset,sent)
            numset.sort(key=lambda x:x[1])
            labels=[]
            j=0
            for i,t in enumerate(tokens):
                while j<len(numset) and numset[j][1]<pmap[i]:
                    j+=1
                if j<len(numset) and numset[j][1]==pmap[i]:
                    labels.append('1')
                else:
                    labels.append('0')
            ofl.write(' '.join(labels)+'\n')
