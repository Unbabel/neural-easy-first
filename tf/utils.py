# coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle as pkl
import codecs
import embedding

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


def load_data(feature_label_file, embedding_src, embedding_tgt, max_sent=0, task="bin", train=False, labeled=True):
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

                feature_vector.append([left_context_id, token_id, right_context_id, src_left_context_id, aligned_token_id, src_right_context_id])
                tgt_sentence.append(token)

                if labeled:
                    # get label
                    label = split_line[-1]
                else:
                    # dummy labels
                    label = "OK"  # TODO
                label_sentence.append(label_dict[label])



    print "Loaded %d sentences" % len(feature_vectors)
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
    window_size = len(X[0][0])
    #print "window size", window_size
    seq_lens = []
    masks = np.zeros(shape=(len(X), max_len))
    i = 0
    X_padded = np.zeros(shape=(len(X), max_len, window_size))
    X_padded.fill(PAD_symbol)
    Y_padded = np.zeros(shape=(len(Y), max_len))
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
    return X_padded, Y_padded, masks, seq_lens


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
    f1_1 = 2 * (precision_1*recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    f1_2 = 2 * (precision_2*recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 else 0
    return f1_1, f1_2




if __name__ == "__main__":

    # test embedding loader
    tgt_embeddings = load_embedding("/home/julia/Dokumente/Unbabel/neural-easy-first/data/WMT2016/embeddings/polyglot-de.pkl")
    src_embeddings = load_embedding("/home/julia/Dokumente/Unbabel/neural-easy-first/data/WMT2016/embeddings/polyglot-en.pkl")


    # test data loader
    data_file = "/home/julia/Dokumente/Unbabel/neural-easy-first/data/WMT2016/WMT2016/task2_en-de_dev/dev.basic_features_with_tags"
    data = load_data(data_file, src_embeddings, tgt_embeddings,  max_sent=10)
    print data

    # test f1 eval
    y = [[0,1,1,1,1,0], [0,1,1,1]]
    y_pred = [[0,0,0,0,1,0], [0,1,1,1]]
    print f1s_binary(y, y_pred)