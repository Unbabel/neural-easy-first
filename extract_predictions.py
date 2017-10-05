import sys

fix_inconsistencies = False

filepath = sys.argv[1]
f = open(filepath)
sentence = []
for line in f:
    line = line.rstrip('\n')
    if line == '':
        words, gold, predicted = sentence[-3:]
        words = words.split(' ')
        gold = gold.split(' ')
        predicted = predicted.split(' ')
        if fix_inconsistencies:
            for i in xrange(len(words)-1, -1, -1):
                p = predicted[i]
                if i < len(words)-1:
                    if predicted[i+1][:2] == 'I-':
                        if p[2:] != predicted[i+1][2:]:
                            predicted[i] = 'B-' + predicted[i+1][2:]
        for w, g, p in zip(words, gold, predicted):
            print ' '.join([w, g, g, p])
        print
        sentence = []
    sentence.append(line)

f.close()
