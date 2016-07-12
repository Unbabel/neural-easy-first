# coding=utf-8
import numpy as np
import tensorflow as tf

def glorot_init(shape, distribution, type):
    """
    Glorot initialization scheme
    Glorot & Bengio: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    :param shape:
    :param distribution:
    :param type:
    :return:
    """
    fan_in, fan_out = shape
    if distribution=="uniform":
        val = np.sqrt(6./(fan_in+fan_out))
        return tf.random_uniform_initializer(dtype=type, minval=-val, maxval=val)
    else:
        val = np.sqrt(2./(fan_in+fan_out))
        return tf.random_normal_initializer(dtype=type, stddev=val)


def pad_data(X, Y, max_len):
    """
    Pad data up till maximum length and create masks and lists of sentence lengths
    :param X:
    :param Y:
    :param max_len:
    :return:
    """
    seq_lens = []
    masks = np.zeros(shape=(len(X), max_len))
    i = 0
    X_padded = np.zeros(shape=(len(X), max_len))
    Y_padded = np.zeros(shape=(len(Y), max_len))

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
