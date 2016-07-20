
class embedding:
    def __init__(self, table, word2id, id2word, UNK_id, PAD_id, end_id, start_id):
        self.table = table
        self.word2id = word2id
        self.id2word = id2word
        self.UNK_id = UNK_id
        self.PAD_id = PAD_id
        self.end_id = end_id
        self.start_id = start_id

    def get_id(self, word):
        return self.word2id.get(word, self.UNK_id)

    def lookup(self, word):
        return self.table[self.word2id(word)]

    def __str__(self):
        return "Embeddings with vocab_size=%d" % (len(self.word2id))