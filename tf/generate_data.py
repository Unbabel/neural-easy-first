import numpy as np
from hmmlearn import hmm
import cPickle as pkl
import argparse

# generate dummy data with hmm for sequence tagging problems


def softmax_row(x, tau=1.):
    """
    softmax by row
    :param x:
    :return:
    """
    softmax_row = []
    for r in xrange(x.shape[0]):
        row = np.exp(x[r]/tau) / np.sum(np.exp(x[r]/tau), axis=0)
        softmax_row.append(row)
    return np.array(softmax_row)


def generate_hmm_data_probs(number_of_labels, vocab_size, dump_prefix):
    """
    Generate the probabilities for an hmm and pickle them
    :param number_of_labels:
    :param vocab_size:
    :param dump_file:
    :return:
    """
    # random start probabilities
    init = np.random.randn(1, number_of_labels)
    init_prob = softmax_row(init, 0.5)[0]
    print "start probs", init_prob

    # random transition probabilities
    a = np.random.randn(number_of_labels, number_of_labels)
    a_prob = softmax_row(a)  # transition probabilities
    print "transition prob", a_prob

    # random output probs
    o = np.random.randn(number_of_labels, vocab_size)
    o_prob = softmax_row(o, number_of_labels*2)  # output probabilities
    print "output prob", o_prob

    # pickle
    dump_file = "%slabels%d.vocab%d.pkl" % (
        dump_prefix, number_of_labels, vocab_size)
    pkl.dump((init_prob, a_prob, o_prob), open(dump_file, "wb"))
    print "dumped to ", dump_file
    return dump_file


def generate_hmm_data_from_fixed(numbers_of_instances, instance_length,
                                 prob_file):
    init_prob, a_prob, o_prob = pkl.load(prob_file)
    number_of_labels = len(init_prob)
    vocab_size = o_prob.shape[1]
    # create hmm
    hmm_model = hmm.MultinomialHMM(number_of_labels)
    hmm_model.transmat_ = a_prob
    hmm_model.startprob_ = init_prob
    hmm_model.emissionprob_ = o_prob
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x, y = [], []
        for i in range(number_of_instances):
            # generate input
            x_i_len = np.random.randint(1, instance_length+1)
            x_i = np.array([[np.random.randint(vocab_size)]
                            for j in range(x_i_len)])
            y_i = hmm_model.predict(x_i)
            # or sample from model directly:
            # x_i, y_i = hmm_model.sample(x_i_len)
            x.append(x_i.flatten())
            y.append(y_i)
        xs.append(x)
        ys.append(y)
    return xs, ys


def write_data_to_file(x, y, output_file):
    with open(output_file, "w") as out_f:
        sent_counter = 0
        word_counter = 0
        for x_sent, y_sent in zip(x, y):
            sent_counter += 1
            assert len(x_sent) == len(y_sent)
            for x_word, y_word in zip(x_sent, y_sent):
                word_counter += 1
                out_f.write(("%d\t%d\n" % (x_word, y_word)))
            out_f.write("\n")
    print("Wrote %d sentences with %d words to %s" %
          (sent_counter, word_counter, output_file))

def main():
    parser = argparse.ArgumentParser(description='Generate random HMM data.')
    parser.add_argument('num_labels', type=int, help='number of labels')
    parser.add_argument('vocab_size', type=int, help='size of vocabulary')
    parser.add_argument('data_path', type=str, help='where to store the data')
    parser.add_argument('max_len', type=int, help='maximum length for sequences')
    parser.add_argument('-n', '--new_prob', action='store_true',
                        help='whether to generate new probabilities '
                             'or just load existing ones')
    parser.add_argument('data_sizes', type=int, nargs=3, metavar='size',
                        help='number of sentences for each dataset '
                             '(train, dev, test')

    args = parser.parse_args()

    # generate or load distributions for hmm model
    if args.new_prob:
        prob_file = generate_hmm_data_probs(args.num_labels, args.vocab_size,
                                            args.data_path)
    else:
        prob_file = "%s.labels%d.vocab%d.pkl" % (
            args.data_path, args.num_labels, args.vocab_size)

    # generate corpus from hmm
    sets = ["train", "dev", "test"]
    print prob_file
    xs, ys = generate_hmm_data_from_fixed(args.data_sizes, args.max_len,
                                          open(prob_file, "rb"))
    out_f = prob_file.split(".pkl")[0]
    for x, y, s in zip(xs, ys, sets):
        set_file = out_f+"."+s+".tagging"
        write_data_to_file(x, y, set_file)
        print "wrote %s data to %s" % (s, set_file)

if __name__ == "__main__":
    main()