import chinese2digits as c2d
import sys


with open(sys.argv[1]) as f, open(sys.argv[2],'w') as fout:
    for i,line in enumerate(f):
        print(i,line)
        output = c2d.takeNumberFromString(line.rstrip().replace(' ',''))
        if len(output['CHNumberStringList']) == 0:
            fout.write('None\n')
        else:
            fout.write(line)
            fout.write(' '.join(output['CHNumberStringList']) + '\n')
            fout.write(' '.join(output['digitsStringList']) + '\n')
            fout.write('\n')
