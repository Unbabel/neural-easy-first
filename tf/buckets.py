import numpy as np
import operator
import logging
import pdb

class PaddedData(object):
    def __init__(self, inputs=None, labels=None, masks=None, lengths=None):
        self.inputs = inputs
        self.labels = labels
        self.masks = masks
        self.lengths = lengths

    def num_sequences(self):
        return self.inputs.shape[0]

    def max_length(self):
        return self.inputs.shape[1]

    def num_features(self):
        return self.inputs.shape[2]

    def select(self, indices):
        return PaddedData(self.inputs[indices,:,:],
                          self.labels[indices,:],
                          self.masks[indices,:],
                          self.lengths[indices])

    def populate(self, input_sequences, label_sequences, max_length,
                 pad_symbol=0):
        num_sequences = len(input_sequences)
        num_features = len(input_sequences[0][0])
        lengths = []
        self.masks = np.zeros(shape=(num_sequences, max_length), dtype=int)
        self.inputs = np.zeros(shape=(num_sequences, max_length, num_features),
                               dtype=int)
        self.inputs.fill(pad_symbol)
        self.labels = np.zeros(shape=(num_sequences, max_length), dtype=int)
        self.labels.fill(pad_symbol)

        i = 0
        for x, y in zip(input_sequences, label_sequences):
            assert len(x) == len(y)
            length = len(x)
            assert length <= max_length
            lengths.append(length)
            for j in xrange(length):
                self.masks[i][j] = 1
                self.inputs[i][j] = x[j]
                self.labels[i][j] = y[j]
            i += 1
        self.lengths = np.asarray(lengths)


class Bucket(object):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices


class BucketFactory(object):
    def __init__(self):
        pass

    def build_buckets(self, input_sequences, label_sequences, num_buckets=20):
        # Sort data by length.
        lengths = [(len(s), i) for i, s in enumerate(input_sequences)]
        sorted_lengths = sorted(lengths, key=operator.itemgetter(0))
        bucket_size = int(np.ceil(len(input_sequences) / float(num_buckets)))
        logging.info("Creating %d buckets of size %d." % (num_buckets,
                                                          bucket_size))
        buckets_data = [sorted_lengths[i:i+bucket_size] \
                        for i in xrange(0, len(sorted_lengths), bucket_size)]
        # Max len of sequence in each bucket.
        bucket_max_lengths = [bucket[-1][0] for bucket in buckets_data]
        logging.info("Bucket max lengths: %s." % str(bucket_max_lengths))

        # Pad and bucket train data.
        buckets = []
        for j, bucket_data in enumerate(buckets_data):
            indices = np.asarray([i for _, i in bucket_data])
            max_length = bucket_max_lengths[j]
            padded_data = PaddedData()
            padded_data.populate(input_sequences[indices],
                                 label_sequences[indices],
                                 max_length)
            buckets.append(Bucket(padded_data, indices))

        return buckets

    def put_in_buckets(self, input_sequences, label_sequences,
                       reference_buckets):
        """
        Given buckets, put the data in those buckets according to their length.
        May create an additional bucket if the largest length exceeds that of
        the last bucket.
        """
        bucket_lengths = [bucket.data.max_length() \
                          for bucket in reference_buckets]
        input_lengths = np.array([len(s) for s in input_sequences], dtype='int')
        max_length = max(input_lengths)
        input_bucket_index = np.digitize(input_lengths, bucket_lengths,
                                         right=True)
        bucket_indices = [[] for j in xrange(len(reference_buckets)+1)]
        for i in xrange(len(input_sequences)):
            bucket_indices[input_bucket_index[i]].append(i)
        # Pad and bucket data.
        buckets = []
        for j in xrange(len(reference_buckets)+1):
            indices = np.asarray(bucket_indices[j])
            if not len(indices):
                buckets.append(None)
                continue
            padded_data = PaddedData()
            if j == len(reference_buckets):
                length = max_length
            else:
                length = reference_buckets[j].data.max_length()
            padded_data.populate(input_sequences[indices],
                                 label_sequences[indices],
                                 length)
            buckets.append(Bucket(padded_data, indices))

        return buckets
