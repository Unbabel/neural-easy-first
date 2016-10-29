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

    def _build_vocabularies(self, filepath, store=False):
        '''Given a dataset file, extracts the vocabularies and optionally
        stores them in a pickle file. Returns each vocabulary as a dictionary.
        This implementation assumes only one vocabulary (for the words).
        Derive this class and override this function for using additional
        vocabularies.'''
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
        vocabularies = [vocabulary]
        return vocabularies

    def _process_sentence(self, sentence_fields, embeddings, label_dict,
                          train=False):
        '''Processes a sentence given a list of sentence fields (one element per
        token containing a list of fields for that token), a list of embeddings
        (just word embeddings in this implementation), and a dictionary of
        labels. Returns two lists of the same size, containing the sentence
        (a list of lists of features) and the sentence labels (a list of label
        identifiers)'''
        assert len(embeddings) == 1
        word_embeddings = embeddings[0]
        sentence = []
        sentence_labels = []
        for fields in sentence_fields:
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
            word_id = word_embeddings.get_id(word)
            sentence.append([word_id])
            sentence_labels.append(label_id)
        return sentence, sentence_labels

    def num_embedding_features(self):
        '''Returns a list whose size is the number of embedding objects, and
        where each element contains the number of features associated to each
        embedding object. In this implementation, it's just one embedding object
        (word embeddings) associated to a single feature (the word).'''
        return [1]

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
            # if embeddings are not given, build vocabularies and
            # prepare the embeddings.
            embeddings = []
            vocabularies = self._build_vocabularies(filepath, store=True)
            for vocabulary in vocabularies:
                word2id = {word: i for word, i in vocabulary.iteritems()}
                id2word = {i: word for word, i in vocabulary.iteritems()}
                embeddings.append(embedding.Embedding(None, word2id, id2word,
                                                      vocabulary["<UNK>"],
                                                      vocabulary["<PAD>"],
                                                      vocabulary["</s>"],
                                                      vocabulary["<s>"]))

        assert len(embeddings) == len(self.num_embedding_features())

        # Load words and labels
        sentences = []
        labels = []
        sentence_fields = []
        unks = set()
        with codecs.open(filepath, "r", "utf8") as data:
            for line in data:
                line = line.rstrip()
                if line == '':  # End of sentence.
                    if max_length < 0 or len(sentence_fields) < max_length:
                        sentence, sentence_labels = self._process_sentence(
                            sentence_fields, embeddings, label_dict, train)
                        assert len(sentence) == len(sentence_labels)
                        assert len(sentence[0]) == \
                            sum(self.num_embedding_features())
                        sentences.append(sentence)
                        labels.append(sentence_labels)
                    sentence_fields = []
                else: # One word per line.
                    fields = line.split("\t")
                    sentence_fields.append(fields)

        logging.info("Loaded %d sentences", len(sentences))
        logging.info("%d UNK words", len(unks))
        if train:
            return sentences, labels, label_dict, embeddings
        else:
            return sentences, labels, label_dict


class QualityDatasetReader(DatasetReader):
    '''A class for reading a word-level quality estimation dataset.'''
    def __init__(self):
        pass

    def _build_vocabularies(self, filepath, store=False):
        '''Given a QE dataset file, extracts the vocabularies and optionally
        stores them in a pickle file. Returns a list of two dictionaries,
        containing the target and source vocabularies respectively.'''
        # TODO(afm): implement cutoff.
        # TODO(afm): allow filtering by sentence length.
        # Initialize vocabulary with special symbols.
        target_vocabulary = {"<UNK>": 0, "<PAD>": 1, "<s>": 2, "</s>": 3}
        source_vocabulary = {"<UNK>": 0, "<PAD>": 1, "<s>": 2, "</s>": 3}
        with codecs.open(filepath, "r", "utf8") as pos_data:
            for line in pos_data:
                line = line.rstrip()
                if line == '':
                    continue
                else:
                    fields = line.split()
                    target_words = [fields[3]]
                    source_words = fields[6:8]
                    for word in target_words:
                        if word not in target_vocabulary:
                            word_id = len(target_vocabulary)
                            target_vocabulary[word] = word_id
                    for word in source_words:
                        if word not in source_vocabulary:
                            word_id = len(source_vocabulary)
                            source_vocabulary[word] = word_id
        logging.info("Built target vocabulary of %d words.",
                     len(target_vocabulary))
        logging.info("Built source vocabulary of %d words.",
                     len(source_vocabulary))
        if store:
            dump_file = filepath + ".vocab.target.pkl"
            pkl.dump(target_vocabulary, open(dump_file, "wb"))
            logging.info("Stored target vocabulary in %s.", dump_file)
            dump_file = filepath + ".vocab.source.pkl"
            pkl.dump(source_vocabulary, open(dump_file, "wb"))
            logging.info("Stored source vocabulary in %s.", dump_file)
        vocabularies = [target_vocabulary, source_vocabulary]
        return vocabularies

    def _process_sentence(self, sentence_fields, embeddings, label_dict,
                          train=False):
        '''Processes a sentence given a list of sentence fields (one element per
        token containing a list of fields for that token), a list of embeddings
        (containing the target and source embeddings), and a dictionary of
        labels. Returns two lists of the same size, containing the sentence
        (a list of lists of features) and the sentence labels (a list of label
        identifiers).'''
        assert len(embeddings) == 2
        target_embeddings = embeddings[0]
        source_embeddings = embeddings[1]
        sentence = []
        sentence_labels = []
        for fields in sentence_fields:
            word = fields[3]
            left_context = fields[4]
            right_context = fields[5]
            source_word = fields[6]
            source_left_context = fields[7]
            source_right_context = fields[8]
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
            word_id = target_embeddings.get_id(word)
            left_context_id = target_embeddings.get_id(left_context)
            right_context_id = target_embeddings.get_id(right_context)
            source_word_id = source_embeddings.get_id(source_word)
            source_left_context_id = \
                source_embeddings.get_id(source_left_context)
            source_right_context_id = \
                source_embeddings.get_id(source_right_context)

            if "|" in source_word:  # Multiple alignments.
                if source_word_id == source_embeddings.unk_id:
                    # This pair of alignments has not been seen during
                    # training.
                    source_words = source_word.split("|")
                    for source_subword in source_words:
                        wid = source_embeddings.get_id(source_subword)
                        if wid != source_embeddings.unk_id:
                            # No averaging during test time possible.
                            source_word_id = wid
            sentence.append([word_id,
                             left_context_id,
                             right_context_id,
                             source_word_id,
                             source_left_context_id,
                             source_right_context_id])
            sentence_labels.append(label_id)
        return sentence, sentence_labels

    def num_embedding_features(self):
        '''Returns a list whose size is the number of embedding objects, and
        where each element contains the number of features associated to each
        embedding object. In this implementation, it's just two embedding
        objects (target and source embeddings) with three features each (word,
        left context, and right context).'''
        return [3, 3]

