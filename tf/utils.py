# coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle as pkl
import codecs
import embedding
from scipy import stats
import operator


def load_embedding(pkl_file):
    word2id = {}
    id2word = {}
    with open(pkl_file, "rb") as opened:
        words, vectors = pkl.load(opened)
        assert len(words) == len(vectors)
        UNK_id = words.index("<UNK>")
        PAD_id = words.index("<PAD>")
        start_id = words.index("<S>")
        end_id = words.index("</S>")
        word2id["<s>"] = start_id
        word2id["</s>"] = end_id
        for i, w in enumerate(words):
            word2id[w] = i
            id2word[i] = w
    print "Loaded embeddings for %d words with dimensionality %d" % (len(words), len(vectors[0]))
    print "Special tokens:", UNK_id, PAD_id, start_id, end_id
    emb = embedding.embedding(vectors, word2id, id2word, UNK_id, PAD_id, end_id, start_id)
    return emb


def load_vocabs(train_src, train_tgt, train_features, src_limit,tgt_limit, freq_limit):
    """
    Load the vocabulary of src and tgt size of a qe corpus
    and limit it to a certain size (by frequency).
    In contrast to 'load_vocabs_from_features' the absolute frequencies are
    not distorted by alignments.
    :param train_src:
    :param train_tgt:
    :param train_features:
    :param src_limit:
    :param tgt_limit:
    :param freq_limit:
    :return:
    """
    src_vocab = {}
    tgt_vocab = {}

    # count occurrences in tgt and src text
    with codecs.open(train_src, "r", "utf8") as src, \
        codecs.open(train_tgt, "r", "utf8") as tgt:
        for src_line in src:
            src_tokens = src_line.split()
            for src_token in src_tokens:
                freq = src_vocab.get(src_token, 0)
                src_vocab[src_token] = freq+1
        for tgt_line in tgt:
            tgt_tokens = tgt_line.split()
            for tgt_token in tgt_tokens:
                freq = tgt_vocab.get(tgt_token, 0)
                tgt_vocab[tgt_token] = freq+1

    # count occurrences of combinations of multiple aligned src words
    with codecs.open(train_features, "r", "utf8") as qe_data:
        for line in qe_data:
            stripped = line.strip()
            if stripped == "":  # sentence end
                continue
            else:
                split_line = stripped.split("\t")
                src_tokens = split_line[6:9]
                for src_token in src_tokens:
                    if "|" in src_token:  # multiple alignments: treat as single and as combined word
                        freq = src_vocab.get(src_token, 0)
                        src_vocab[src_token] = freq + 1

     # sort by frequency (descending)
    src_vocab_by_freq = sorted(src_vocab.items(), key=operator.itemgetter(1), reverse=True)
    tgt_vocab_by_freq = sorted(tgt_vocab.items(), key=operator.itemgetter(1), reverse=True)

    print "found %d tokens in train src vocabulary" % len(src_vocab_by_freq)
    print "found %d tokens in train tgt vocabulary" % len(tgt_vocab_by_freq)

    #cut off infrequent ones
    if src_limit > 0:
        src_vocab_by_freq = src_vocab_by_freq[:src_limit]
        print "cutting down src vocab to %d tokens" % len(src_vocab_by_freq)
    if tgt_limit > 0:
        tgt_vocab_by_freq = tgt_vocab_by_freq[:tgt_limit]
        print "cutting down tgt vocab to %d tokens" % len(tgt_vocab_by_freq)
    if freq_limit > 0:
        src_vocab_by_freq = [(word, freq) for (word, freq) in src_vocab_by_freq if freq > freq_limit]
        tgt_vocab_by_freq = [(word, freq) for (word, freq) in tgt_vocab_by_freq if freq > freq_limit]

    vocab = {"<PAD>", "<UNK>", "<s>", "</s>"}
    src_words = list(set(map(operator.itemgetter(0), src_vocab_by_freq)).union(vocab))
    tgt_words = list(set(map(operator.itemgetter(0), tgt_vocab_by_freq)).union(vocab))

    print "final %d tokens in src vocabulary" % len(src_words)
    print "final %d tokens in tgt vocabulary" % len(tgt_words)

    return src_words, tgt_words


def load_vocabs_from_features(feature_file, src_limit=0, tgt_limit=0):
    """
    Load the vocabulary of a qe corpus (in "feature and tags file")
    and limit it to a certain size (by frequency)
    :param feature_file:
    :param src_limit:
    :param tgt_limit:
    :return:
    """
    src_vocab = {}
    tgt_vocab = {}
    with codecs.open(feature_file, "r", "utf8") as qe_data:

        # collect all tokens and count them
        for line in qe_data:
            stripped = line.strip()
            if stripped == "":  # sentence end
                continue
            else:
                split_line = stripped.split("\t")
                tgt_tokens = split_line[3]
                src_tokens = split_line[6:9]
                for src_token in src_tokens:
                    if src_token == " ":
                        src_token = "</s>"  # FIXME because of data format (end of src sentence=space)
                    if "|" in src_token:  # multiple alignments: treat as single and as combined word
                        sub_tokens = src_token.split("|")
                        for sub_token in sub_tokens:
                            freq = src_vocab.get(sub_token, 0)
                            src_vocab[sub_token] = freq + 1
                    freq = src_vocab.get(src_token, 0)
                    src_vocab[src_token] = freq + 1
                for tgt_token in tgt_tokens:
                    freq = tgt_vocab.get(tgt_token, 0)
                    tgt_vocab[tgt_token] = freq + 1

    # sort by frequency (descending)
    src_vocab_by_freq = sorted(src_vocab.items(), key=operator.itemgetter(1), reverse=True)
    tgt_vocab_by_freq = sorted(tgt_vocab.items(), key=operator.itemgetter(1), reverse=True)

    print "found %d tokens in train src vocabulary" % len(src_vocab_by_freq)
    print "found %d tokens in train tgt vocabulary" % len(tgt_vocab_by_freq)

    #cut off infrequent ones
    if src_limit > 0:
        src_vocab_by_freq = src_vocab_by_freq[:src_limit]
        print "cutting down src vocab to %d tokens" % len(src_vocab_by_freq)
    if tgt_limit > 0:
        tgt_vocab_by_freq = tgt_vocab_by_freq[:tgt_limit]
        print "cutting down tgt vocab to %d tokens" % len(tgt_vocab_by_freq)

    vocab = {"<PAD>", "<UNK>", "<s>", "</s>"}
    src_words = list(set(map(operator.itemgetter(0), src_vocab_by_freq)).union(vocab))
    tgt_words = list(set(map(operator.itemgetter(0), tgt_vocab_by_freq)).union(vocab))

    print "final %d tokens in src vocabulary" % len(src_words)
    print "final %d tokens in tgt vocabulary" % len(tgt_words)

    return src_words, tgt_words


def build_vocab(feature_file, origin, store=False):
    # FIXME not very efficient
    vocab = ["<PAD>", "<UNK>", "<s>", "</s>"]
    with codecs.open(feature_file, "r", "utf8") as qe_data:
        for line in qe_data:
            stripped = line.strip()
            if stripped == "":  # sentence end
                continue
            else:
                split_line = stripped.split()
                if origin == "tgt":
                    tokens = [split_line[3]]
                else:
                    tokens = split_line[6:8]
                for token in tokens:
                    if token not in vocab:
                        vocab.append(token)
    print "Built %s vocabulary of %d words" % (origin, len(vocab))
    if store:
        dump_file = feature_file+".vocab."+origin+".pkl"
        pkl.dump(vocab, open(dump_file, "wb"))
        print "Stored %s vocabulary in %s" % (origin, dump_file)
    return vocab, 0, 1, 2, 3


def load_data(feature_label_file, embedding_src, embedding_tgt, max_sent=0, task="bin", train=False,
              labeled=True):
    """
    Given a dataset file with features and labels, and word embeddings, read them to lists and dictionaries
    :param feature_label_file:
    :param embedding_src:
    :param embedding_tgt:
    :param max_sent:
    :param task:
    :return:
    """
    # input:
    # - file with features and labels for each tgt word
    #   format (basic features with tags): 30      26      1.15384615385   Die     <s>     Dateien The     <s>     files   0       0       0       0       0       0       0       0       0       0       0       _       _       OK
    # - source, target embeddings (pkl)
    #   format: word array: loaded[0], embeddings: loaded[1]

    # TODO specify feature config for feature selection

    # label dictionary
    label_dict = {}
    if task == "bin":
        label_dict["OK"] = 0
        label_dict["BAD"] = 1
    else:
        raise ValueError("can't handle this task yet, only binary labels")

    if embedding_src is None:
        # if embeddings are not given, build vocabulary  # TODO specify vocab size
        vocab, UNK_id, PAD_id, start_id, end_id = build_vocab(feature_label_file, "src", True)
        word2id = {word: i for i, word in enumerate(vocab)}
        id2word = {i: word for i, word in enumerate(vocab)}
        embedding_src = embedding.embedding(None, word2id, id2word, UNK_id, PAD_id, end_id, start_id)

    if embedding_tgt is None:
        # if embeddings are not given, build vocabulary
        vocab, UNK_id, PAD_id, start_id, end_id = build_vocab(feature_label_file, "tgt", True)
        word2id = {word: i for i, word in enumerate(vocab)}
        id2word = {i: word for i, word in enumerate(vocab)}
        embedding_tgt = embedding.embedding(None, word2id, id2word, UNK_id, PAD_id, end_id, start_id)


    # load features and labels
    feature_vectors = []
    feature_vector = []
    tgt_sentences = []
    tgt_sentence = []
    labels = []
    label_sentence = []
    tgt_unks = set()
    src_unks = set()
    with codecs.open(feature_label_file, "r", "utf8") as qe_data:
        for line in qe_data:
            stripped = line.strip()
            if stripped == "":  # sentence end
                feature_vectors.append(feature_vector)
                assert len(tgt_sentence) == len(label_sentence)
                tgt_sentences.append(tgt_sentence)
                labels.append(label_sentence)
                label_sentence = []
                feature_vector = []
                tgt_sentence = []
                if len(feature_vectors) >= max_sent and max_sent > 0:
                    break
            else:  # one word per line
                split_line = stripped.split("\t")

                # select features
                token = split_line[3]
                left_context = split_line[4]
                right_context = split_line[5]
                aligned_token = split_line[6]
                src_left_context = split_line[7]
                src_right_context = split_line[8]

                # lookup features
                token_id = embedding_tgt.get_id(token)
                left_context_id = embedding_tgt.get_id(left_context)
                right_context_id = embedding_tgt.get_id(right_context)
                src_left_context_id = embedding_src.get_id(src_left_context)
                src_right_context_id = embedding_src.get_id(src_right_context)
                aligned_token_id = embedding_src.get_id(aligned_token)

                if "|" in aligned_token:  # multiple alignments
                    if aligned_token_id == embedding_src.UNK_id: # this pair of alignments has not been seen during training
                        aligned_tokens = aligned_token.split("|")
                        for at in aligned_tokens:
                            at_id = embedding_src.get_id(at)
                            if at_id != embedding_src.UNK_id:
                                aligned_token_id = at_id  # no averaging during test time possible

                # keep track of unknown words
                if token_id == embedding_tgt.UNK_id:
                    tgt_unks.add(token)
                if left_context_id == embedding_tgt.UNK_id:
                    tgt_unks.add(left_context)
                if right_context_id == embedding_tgt.UNK_id:
                    tgt_unks.add(right_context)
                if src_left_context_id == embedding_src.UNK_id:
                    src_unks.add(src_left_context)
                if src_right_context_id == embedding_src.UNK_id:
                    src_unks.add(src_right_context)
                if aligned_token_id == embedding_src.UNK_id:
                    src_unks.add(aligned_token)

                feature_vector.append([left_context_id, token_id, right_context_id,
                                       src_left_context_id, aligned_token_id, src_right_context_id])
                tgt_sentence.append(token)

                if labeled:
                    # get label
                    label = split_line[-1]
                else:
                    # dummy labels
                    label = "OK"
                label_sentence.append(label_dict[label])

    print "Loaded %d sentences" % len(feature_vectors)
    print "%d UNK words in src" % len(src_unks)
    print "%d UNK words in tgt" % len(tgt_unks)
    if train:
        return feature_vectors, tgt_sentences, labels, label_dict, embedding_src, embedding_tgt
    else:
        return feature_vectors, tgt_sentences, labels, label_dict


def pad_data(X, Y, max_len, PAD_symbol=0):
    """
    Pad data up till maximum length and create masks and lists of sentence lengths
    :param X:
    :param Y:
    :param max_len:
    :return:
    """
    #print "to pad", X[0], Y[0]
    feature_size = len(X[0][0])
    #print "feature size", feature_size
    seq_lens = []
    masks = np.zeros(shape=(len(X), max_len), dtype=int)
    i = 0
    X_padded = np.zeros(shape=(len(X), max_len, feature_size), dtype=int)
    X_padded.fill(PAD_symbol)
    Y_padded = np.zeros(shape=(len(Y), max_len), dtype=int)
    Y_padded.fill(PAD_symbol)

    for x, y in zip(X, Y):
        assert len(x) == len(y)
        seq_len = len(x)
        if seq_len > max_len:
            seq_len = max_len
        seq_lens.append(seq_len)
        for j in range(seq_len):
            masks[i][j] = 1
            X_padded[i][j] = x[j]
            Y_padded[i][j] = y[j]
        i += 1
    #print "padded", X_padded[0], seq_lens[0]
    return X_padded, Y_padded, masks, np.asarray(seq_lens)


def buckets_by_length(data, labels, buckets=20, max_len=0, mode='pad'):
    """
    :param data: numpy arrays of data
    :param labels: numpy arrays of labels
    :param buckets: list of buckets (lengths) into which to group samples according to their length.
    :param mode: either 'truncate' or 'pad':
                * When truncation, remove the final part of a sample that does not match a bucket length;
                * When padding, fill in sample with zeros up to a bucket length.
                The obvious consequence of truncating is that no sample will be padded.
    :return: a dictionary of grouped data and a dictionary of the data original indexes, both keyed by bucket, and the bin edges
    """
    input_lengths = np.array([len(s) for s in data], dtype='int')  # for dev and train (if dev given)

    maxlen = max_len if max_len > 0 else max(input_lengths) + 1

    # sort data by length
    # split this array into 'bucket' many parts, these are the buckets
    data_lengths_with_idx = [(len(s), i) for i, s in enumerate(data)]
    sorted_data_lengths_with_idx = sorted(data_lengths_with_idx, key=operator.itemgetter(0))
    bucket_size = int(np.ceil(len(data)/float(buckets)))
    print "Creating %d Buckets of size %f" % (buckets, bucket_size)
    buckets_data = [sorted_data_lengths_with_idx[i:i+bucket_size] for i in xrange(0, len(sorted_data_lengths_with_idx), bucket_size)]
    bin_edges = [bucket[-1][0] for bucket in buckets_data]  # max len of sequence in bucket
    print "bin_edges", bin_edges
    if bin_edges[-1] < maxlen:
        bin_edges[-1] = maxlen
    print "final bin_edges", bin_edges
    input_bucket_index = np.zeros(shape=len(data), dtype=int)
    for bucket_idx, bucket in enumerate(buckets_data):
        for l, d_idx in bucket:
            input_bucket_index[d_idx] = bucket_idx

    # pad and bucket train data
    bucketed_data = {}
    reordering_indexes_train = {}
    for bucket in list(np.unique(input_bucket_index)):
        length_indexes = np.where(input_bucket_index == bucket)[0]
        reordering_indexes_train[bucket] = length_indexes
        maxlen = bin_edges[bucket]
        padded_data = pad_data(data[length_indexes], labels[length_indexes], max_len=maxlen)
        bucketed_data[bucket] = padded_data # in final dict, start counting by zero

    return bucketed_data, reordering_indexes_train, bin_edges


def put_in_buckets(data_array, labels, buckets, mode='pad'):
    """
    Given bucket edges and data, put the data in buckets according to their length
    :param data_array:
    :param labels:
    :param buckets:
    :return:
    """
    input_lengths = np.array([len(s) for s in data_array], dtype='int')
    input_bucket_index = [i if i<len(buckets) else len(buckets)-1 for i in np.digitize(input_lengths, buckets, right=False)]  # during testing, longer sentences are just truncated
    if mode == 'truncate':
        input_bucket_index -= 1
    bucketed_data = {}
    reordering_indexes = {}
    for bucket in list(np.unique(input_bucket_index)):
        length_indexes = np.where(input_bucket_index == bucket)[0]
        reordering_indexes[bucket] = length_indexes
        maxlen = int(np.floor(buckets[bucket]))
        padded = pad_data(data_array[length_indexes], labels[length_indexes], max_len=maxlen)
        bucketed_data[bucket] = padded  # in final dict, start counting by zero
    return bucketed_data, reordering_indexes


def accuracy(y_i, predictions):
    """
    Accuracy of word predictions
    :param y_i:
    :param predictions:
    :return:
    """
    assert len(y_i) == len(predictions)
    correct_words, all = 0.0, 0.0
    for y, y_pred in zip(y_i, predictions):
        # predictions can be shorter than y, because inputs are cropped to specified maximum length
        for y_w, y_pred_w in zip(y, y_pred):
            all += 1
            if y_pred_w == y_w:
                correct_words += 1
    return correct_words/all


def f1s_binary(y_i, predictions):
    """
    F1 scores of two-class predictions
    :param y_i:
    :param predictions:
    :return: F1_class1, F1_class2
    """
    assert len(y_i) == len(predictions)
    fp_1 = 0.0
    tp_1 = 0.0
    fn_1 = 0.0
    tn_1 = 0.0
    for y, y_pred in zip(y_i, predictions):
        for y_w, y_pred_w in zip(y, y_pred):
            if y_w == 0:  # true class is 0
                if y_pred_w == 0:
                    tp_1 += 1
                else:
                    fn_1 += 1
            else:  # true class is 1
                if y_pred_w == 0:
                    fp_1 += 1
                else:
                    tn_1 += 1
    tn_2 = tp_1
    fp_2 = fn_1
    fn_2 = fp_1
    tp_2 = tn_1
    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0
    precision_2 = tp_2 / (tp_2 + fp_2) if (tp_2 + fp_2) > 0 else 0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
    recall_2 = tp_2 / (tp_2 + fn_2) if (tp_2 + fn_2) > 0 else 0
    f1_1 = 2 * (precision_1*recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 \
        else 0
    f1_2 = 2 * (precision_2*recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 \
        else 0
    return f1_1, f1_2


if __name__ == "__main__":

    # test embedding loader
    tgt_embeddings = load_embedding("../data/WMT2016/embeddings/polyglot-de.pkl")
    src_embeddings = load_embedding("../data/WMT2016/embeddings/polyglot-en.pkl")

    # test data loader
    data_file = "../data/WMT2016/WMT2016/task2_en-de_dev/dev.basic_features_with_tags"
    data = load_data(data_file, src_embeddings, tgt_embeddings,  max_sent=100)
    print data

    # test padding and bucketing
    feature_vectors, tgt_sentences, labels, label_dict = data
    #print pad_data(feature_vectors, labels, max_len=30)

    bucketed_data, reordering_indexes, bucket_edges = buckets_by_length([np.asarray(feature_vectors)],
                                                          [np.asarray(labels)], buckets=3, mode="pad")
    #print "bucketed data", bucketed_data  # X_padded, Y_padded, masks, seq_lens
    print "reordering idx", reordering_indexes
    print "bucket edges", bucket_edges

    # test putting in pre-defined buckets
    bucketed_data_2, reordering_indexes_2 = put_in_buckets(np.asarray(feature_vectors), np.asarray(labels), buckets=bucket_edges)
    #print "bucketed data (2)", bucketed_data_2
    print "reordering idx (2)", reordering_indexes_2

    # test equal strategy
    bucketed_data, reordering_indexes, bucket_edges = buckets_by_length([np.asarray(feature_vectors)],
                                                          [np.asarray(labels)], buckets=3, mode="pad", strategy="equal")
    #print "bucketed data", bucketed_data  # X_padded, Y_padded, masks, seq_lens
    print "reordering idx", reordering_indexes
    print "bucket edges", bucket_edges

    bucketed_data_2, reordering_indexes_2 = put_in_buckets(np.asarray(feature_vectors), np.asarray(labels), buckets=bucket_edges)
    #print "bucketed data (3)", bucketed_data_2
    print "reordering idx (3)", reordering_indexes_2


    # test f1 eval
    y = [[0,1,1,1,1,0], [0,1,1,1]]
    y_pred = [[0,0,0,0,1,0], [0,1,1,1]]
    print f1s_binary(y, y_pred)
