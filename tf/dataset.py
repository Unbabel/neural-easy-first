# -*- coding: utf-8 -*-
'''This module handles dataset reading and preprocessing.'''

import embedding
import codecs
import logging
import cPickle as pkl

class DatasetReader(object):
    '''A class for reading a sequence tagging dataset.'''
    def __init__(self):
        pass

    def _build_vocabulary(self, filepath, store=False):
        '''Given a dataset file, extracts the vocabulary and optionally
        stores it in a pickle file. Returns the vocabulary as a dictionary.'''
        # TODO(afm): implement cutoff.
        # TODO(afm): allow filtering by sentence length.
        # Initialize vocabulary with special symbols.
        vocabulary = {"<UNK>": 0, "<PAD>": 1, "<s>": 2, "</s>": 3}
        with codecs.open(filepath, "r", "utf8") as pos_data:
            for line in pos_data:
                line = line.rstrip()
                if line == '':
                    continue
                else:
                    fields = line.split()
                    word = fields[0]
                    if word not in vocabulary:
                        word_id = len(vocabulary)
                        vocabulary[word] = word_id
        logging.info("Built vocabulary of %d words.", len(vocabulary))
        if store:
            dump_file = filepath + ".vocab.pkl"
            pkl.dump(vocabulary, open(dump_file, "wb"))
            logging.info("Stored vocabulary in %s.", dump_file)
        return vocabulary

    def load_embeddings(self, pkl_file):
        '''Loads an embedding file, returning an Embedding object.'''
        word2id = {}
        id2word = {}
        with open(pkl_file, "rb") as opened:
            words, vectors = pkl.load(opened)
            assert len(words) == len(vectors)
            unk_id = words.index("<UNK>")
            pad_id = words.index("<PAD>")
            start_id = words.index("<S>")
            end_id = words.index("</S>")
            word2id["<s>"] = start_id
            word2id["</s>"] = end_id
            for i, word in enumerate(words):
                word2id[word] = i
                id2word[i] = word
        logging.info("Loaded embeddings for %d words with dimensionality %d",
                     len(words), len(vectors[0]))
        embeddings = embedding.Embedding(vectors, word2id, id2word,
                                         unk_id, pad_id, end_id, start_id)
        return embeddings

    def load_data(self, filepath, embeddings=None, max_length=-1,
                  label_dict=None, train=False):
        '''Reads a dataset from a file. Creates embeddings if they don't exist
        already. If max_length >= 0, discards sentences longer than max_length.
        Returns a list of sentences (each being a list of word IDs looked up
        from the embeddings' vocabulary), label sequences, and the label
        dictionary. If "train" is True, adds new labels to "label_dict", and
        returns also the embeddings.'''
        if label_dict is None:
            label_dict = {}
        if embeddings is None:
            # if embeddings are not given, build vocabulary.
            vocabulary = self._build_vocabulary(filepath, store=True)
            word2id = {word: i for word, i in vocabulary.iteritems()}
            id2word = {i: word for word, i in vocabulary.iteritems()}
            embeddings = embedding.Embedding(None, word2id, id2word,
                                             vocabulary["<UNK>"],
                                             vocabulary["<PAD>"],
                                             vocabulary["</s>"],
                                             vocabulary["<s>"])
        # Load words and labels
        sentences = []
        sentence = []
        sentence_labels = []
        labels = []
        unks = set()
        with codecs.open(filepath, "r", "utf8") as data:
            for line in data:
                line = line.rstrip()
                if line == '':  # End of sentence.
                    assert len(sentence) == len(sentence_labels)
                    if max_length < 0 or len(sentence) < max_length:
                        sentences.append(sentence)
                        labels.append(sentence_labels)
                    sentence = []
                    sentence_labels = []
                else: # One word per line.
                    fields = line.split("\t")
                    word = fields[0]
                    label = fields[-1]
                    if label not in label_dict:
                        if train:
                            label_id = len(label_dict)
                            label_dict[label] = label_id
                        else:
                            # We don't allow unknown labels at test time.
                            assert False
                    else:
                        label_id = label_dict[label]

                    # Lookup features.
                    word_id = embeddings.get_id(word)

                    # Keep track of unknown words.
                    if word_id == embeddings.unk_id:
                        unks.add(word)

                    sentence.append([word_id])
                    sentence_labels.append(label_id)

        logging.info("Loaded %d sentences", len(sentences))
        logging.info("%d UNK words", len(unks))
        if train:
            return sentences, labels, label_dict, embeddings
        else:
            return sentences, labels, label_dict
