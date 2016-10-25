# -*- coding: utf-8 -*-
'''This module handles loading and storage of word embeddings.'''

import numpy as np
import operator
import cPickle as pkl

class Embedding(object):
    '''A class for handling word embeddings.'''
    def __init__(self, table, word2id, id2word,
                 unk_id, pad_id, end_id, start_id):
        self.table = table
        self.word2id = word2id
        self.id2word = id2word
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.end_id = end_id
        self.start_id = start_id
        self.added_words = 0
        self.multiple_aligned_words = 0

    def vocabulary_size(self):
        '''Returns the number of words in the vocabulary.'''
        return len(self.word2id)

    def get_id(self, word):
        '''Given a word, return its id or the UNK id if the word is not in the
        vocabulary.'''
        return self.word2id.get(word, self.unk_id)

    def lookup(self, word):
        '''Given a word, return its embedding. Handles unknown words.'''
        return self.table[self.word2id(word)]

    def add_word(self, word):
        '''Add a word to an existing embedding and set its embedding to zero.'''
        if word not in self.word2id.keys():
            assert self.table is not None
            new_v = np.zeros_like(self.table[0])
            # Lookup/add single words, then take their avg to initialize
            # combination.
            if "|" in word:
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
            new_id = self.table.shape[0]
            self.word2id[word] = new_id
            self.id2word[new_id] = word
            self.table = np.append(self.table, [new_v], axis=0)
        else:
            new_id = self.word2id[word]
            #print "word exists", word, new_id
        return new_id

    def store(self, filepath):
        '''Dump embeddings to a pickle file.'''
        # Sort entries by id, several words can have the same id
        # (e.g. <s> and <S>).
        sorted_entries = sorted(self.word2id.items(),
                                key=operator.itemgetter(1))
        sorted_words, sorted_ids = zip(*sorted_entries)
        vectors = self.table[np.array(sorted_ids)]
        assert len(sorted_words) == len(vectors)
        pkl.dump((sorted_words, vectors), open(filepath, "wb"))
        print "Dumped embedding to file %s" % filepath

    def __str__(self):
        return "Embeddings with vocab_size=%d" % (len(self.word2id))
