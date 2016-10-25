# -*- coding: utf-8 -*-
'''This module evaluates a sequence tagger.'''

class Evaluator(object):
    '''A class for evaluating a sequence tagger.'''
    def __init__(self):
        pass

    def accuracy(self, y_i, predictions):
        """
        Accuracy of word predictions.
        :param y_i:
        :param predictions:
        :return: accuracy
        """
        assert len(y_i) == len(predictions)
        correct_words, total = 0.0, 0.0
        for y, y_pred in zip(y_i, predictions):
            # Predictions can be shorter than y, because inputs are cropped to
            # specified maximum length.
            for y_w, y_pred_w in zip(y, y_pred):
                total += 1
                if y_pred_w == y_w:
                    correct_words += 1
        return correct_words/total

    def f1s_binary(self, y_i, predictions):
        """
        F1 scores of two-class predictions.
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
        f1_1 = 2 * (precision_1*recall_1) / (precision_1 + recall_1) \
               if (precision_1 + recall_1) > 0 else 0
        f1_2 = 2 * (precision_2*recall_2) / (precision_2 + recall_2) \
               if (precision_2 + recall_2) > 0 else 0
        return f1_1, f1_2
