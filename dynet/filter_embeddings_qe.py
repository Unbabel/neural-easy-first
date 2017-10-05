import sys
import cPickle as pickle

embeddings_pickle_file = sys.argv[1]
dataset_file = sys.argv[2]

sep = '\t'
f = open(dataset_file)
word_list = set()
for line in f:
    line = line.rstrip('\n')
    if line == '':
        continue
    fields = line.split(sep)
    word_list.update(set(fields[3:9]))
f.close()
print >> sys.stderr, '%d words in white list.' % len(word_list)

num_words = 0
f = open(embeddings_pickle_file)
words, vectors = pickle.load(f)
#import pdb; pdb.set_trace()
for w, v in zip(words, vectors):
    #import pdb; pdb.set_trace()
    w = w.encode('utf8')
    if w in word_list:
        num_words += 1
        print '%s %s' % (w, ' '.join([str(val) for val in v]))
f.close()
print >> sys.stderr, '%d of those words have embeddings.' % num_words
