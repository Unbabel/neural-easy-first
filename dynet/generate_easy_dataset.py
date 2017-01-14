import numpy as np
import pdb

num_words = 10000
num_labels = 5
num_train_sentences = 100000
num_dev_sentences = 2000
num_test_sentences = 2000
sentence_length = 15

assert num_words % num_labels == 0

def sample_multinomial(p):
    s = np.random.rand()
    return np.random.choice(len(p), 1, p=p)[0]

def generate_biased_multinomial(size, selected, ratio):
    p = np.zeros(size)
    num_selected = len(selected)
    num_non_selected = size - num_selected
    p[:] = 1./(ratio*num_selected + num_non_selected)
    p[selected] = ratio/(ratio*num_selected + num_non_selected)
    return p

np.random.seed(42)

# Define model.
num_segments = 1
# Generate easy label distribution.
easy_label_distribution = generate_biased_multinomial(num_labels, [0], ratio=10.)

ratio_transitions = 20.
transition_probabilities = np.zeros((num_labels, num_labels))
for i in xrange(num_labels):
    if i == num_labels-1:
        j = 0
    else:
        j = i+1
    transition_probabilities[:, i] = \
        generate_biased_multinomial(num_labels,
                                    [j],
                                    ratio_transitions)

ratio_emissions_hard = 10.
ratio_emissions_easy = 50.
emission_probabilities = np.zeros((num_words, num_labels))
for i in xrange(num_labels):
    if i == 0:
        ratio_emissions = ratio_emissions_easy
    else:
        ratio_emissions = ratio_emissions_hard
    selected_words = np.arange(i*num_words/num_labels,
                               (i+1)*num_words/num_labels)
    emission_probabilities[:, i] = \
        generate_biased_multinomial(num_words,
                                    selected_words,
                                    ratio_emissions)
# Generate data.
for split, num_sentences in zip(['train', 'dev', 'test'],
                                [num_train_sentences,
                                 num_dev_sentences,
                                 num_test_sentences]):
    f = open('easy_synthetic_%s.tagging' % split, 'w')
    for i in xrange(num_sentences):
        split_points = np.arange(1, sentence_length)
        np.random.shuffle(split_points)
        split_points = np.sort(split_points[:(num_segments-1)])
        segments = []
        position = 0
        for j in split_points:
            segments.append((position, j))
            position = j
        segments.append((j, sentence_length))
        words = [-1] * sentence_length
        labels = [-1] * sentence_length
        for segment in segments:
            head = segment[0] + np.random.choice(segment[1]-segment[0], 1)[0]
            #print head, segment
            # Generate head tag and word.
            label = sample_multinomial(easy_label_distribution)
            w = sample_multinomial(emission_probabilities[:, label])
            words[head] = w
            labels[head] = label
            # Generate left tags and words.
            y = label
            for k in xrange(head-1, segment[0]-1, -1):
                y = sample_multinomial(transition_probabilities[:, y])
                w = sample_multinomial(emission_probabilities[:, y])
                words[k] = w
                labels[k] = y
            # Generate right tags and words.
            y = label
            for k in xrange(head+1, segment[1]):
                y = sample_multinomial(transition_probabilities[:, y])
                w = sample_multinomial(emission_probabilities[:, y])
                words[k] = w
                labels[k] = y
        for w, y in zip(words, labels):
            f.write('%s\t%s\n' % (str(w), str(y)))
        f.write('\n')
    f.close()
