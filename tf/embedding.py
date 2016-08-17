import numpy as np

class embedding:
    def __init__(self, table, word2id, id2word, UNK_id, PAD_id, end_id, start_id):
        self.table = table
        self.word2id = word2id
        self.id2word = id2word
        self.UNK_id = UNK_id
        self.PAD_id = PAD_id
        self.end_id = end_id
        self.start_id = start_id
        self.added_words = 0
        self.multiple_aligned_words = 0

    def get_id(self, word):
        return self.word2id.get(word, self.UNK_id)

    def lookup(self, word):
        return self.table[self.word2id(word)]

    def add_word(self,word):
        """
        add a word to an existing embedding and set its embedding to zero
        :param word:
        :return:
        """
        if word not in self.word2id.keys():
            assert self.table is not None
            new_v = np.zeros_like(self.table[0])
            if "|" in word:  # lookup/add single words, then take their avg to initialize combination
                #print "Found multiple aligned words: %s" % word
                self.multiple_aligned_words += 1
                words = word.split("|")
                avg_v = np.zeros_like(self.table[0])
                for w in words:
                    # if not in vocab, add - else just return id and get vector
                    new_id_w = self.add_word(w)
                    v_w = self.table[new_id_w]
                    avg_v += v_w
                avg_v /= len(words)
                new_v = avg_v
            # add new vocab item for this combination
            #print "Adding word %s to vocabulary" % word
            self.added_words += 1
            new_id = len(self.word2id.keys())
            self.word2id[word] = new_id
            self.id2word[new_id] = word
            self.table = np.append(self.table, new_v)
        else:
            new_id = self.word2id[word]
        return new_id

    def __str__(self):
        return "Embeddings with vocab_size=%d" % (len(self.word2id))