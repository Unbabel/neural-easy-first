import sys

filepath = sys.argv[1]
f = open(filepath)
previous_tag = 'O'
#i = 0
for line in f:
    line = line.rstrip('\n')
    if line == '':
        previous_tag = 'O'
        print line
        #print i
        #i = 0
    else:
        fields = line.split(' ')
        tag = fields[-1]
        if tag[:2] == 'I-' and previous_tag not in ['B-' + tag[2:], 'I-' + tag[2:]]:
            tag = 'B-' + tag[2:]
        previous_tag = tag
        fields[-1] = tag
        print ' '.join(fields)
        #i += 1
f.close()
