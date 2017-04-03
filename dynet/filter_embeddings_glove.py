import sys

embeddings_file = sys.argv[1]
dataset_file = sys.argv[2]
lower = True

sep = ' ' # '\t'
f = open(dataset_file)
word_list = set()
for line in f:
    line = line.rstrip('\n')
    if line == '':
        continue
    fields = line.split(sep)
    if lower:
        word_list.add(fields[0].lower())
    else:
        word_list.add(fields[0])
f.close()
print >> sys.stderr, '%d words in white list.' % len(word_list)

num_words = 0
f = open(embeddings_file)
for line in f:
    line = line.rstrip('\n')
    fields = line.split(' ')
    w = fields[0]
    if lower:
        assert w.lower() == w
    #import pdb; pdb.set_trace()
    if w in word_list:
        num_words += 1
        print '%s %s' % (w, ' '.join([val for val in fields[1:]]))
print >> sys.stderr, '%d of those words have embeddings.' % num_words
