import sys
import numpy as np
import operator
import pdb

def read_dataset(fname, maximum_sentence_length=-1):
    sent = []
    for line in file(fname):
        line = line.strip().split()
        if not line:
            if sent and (maximum_sentence_length < 0 or
                         len(sent) < maximum_sentence_length):
                yield sent
            sent = []
        else:
            w, t = line
            sent.append((w, t))

def create_word_tag_dictionary(instances, smoothing=1.):
    word_tag_dictionary = {}
    tags = set()
    for instance in instances:
        for w, t in instance:
            if w not in word_tag_dictionary:
                word_tag_dictionary[w] = {}
            if t not in word_tag_dictionary[w]:
                word_tag_dictionary[w][t] = 1.
            else:
                word_tag_dictionary[w][t] += 1.
            tags.add(t)
    num_tags = len(tags)
    word_entropies = {}
    for word in word_tag_dictionary:
        word_tags = word_tag_dictionary[word]
        values = np.array(word_tags.values())
        probs = np.zeros(num_tags) + smoothing
        probs[np.arange(len(values))] = values
        probs /= probs.sum()
        word_entropies[word] = -sum(probs * np.log(probs))
    return word_tag_dictionary, tags, word_entropies

def compute_easy_first_ordering(instances, word_entropies, filepath,
                                easy_list_size=100):
    f = open(filepath, 'w')
    min_segment_length = 4
    sorted_word_entropies = sorted(word_entropies.items(),
                                   key=operator.itemgetter(1))
    easy_list = set([w for w, _ in sorted_word_entropies[:easy_list_size]])
    for instance in instances:
        # Get words' entropies.
        words = [w for w, _ in instance]
        entropies = [word_entropies[w] if w in word_entropies else np.inf for w, _ in instance]
        segments = []
        ind = np.argsort(entropies)
        index = ind[0]
        segments.append([index, index+1, index])
        for index in ind[1:]:
            if words[index] in easy_list:
                segments.append([index, index+1, index])
            else:
                break
        heads = [s[-1] for s in segments]
        ordered_segments = [heads.index(h) for h in sorted(heads)]
        previous_segment = segments[ordered_segments[0]]
        previous_segment[0] = 0 # Start at beginning of sentence.
        for k in ordered_segments[1:]:
            # Find split point between the previous segment and the current one.
            current_segment = segments[k]
            while True:
                previous_position = previous_segment[1]
                current_position = current_segment[0]-1
                if current_position+1 == previous_position:
                    break
                assert len(words) > current_position >= previous_position >= 0, pdb.set_trace()
                if entropies[current_position] < entropies[previous_position]:
                    current_segment[0] -= 1
                else:
                    previous_segment[1] += 1
            if previous_segment[1] - previous_segment[0] <= min_segment_length:
                # Merge segments with less than 2 words.
                current_segment[0] = previous_segment[0]
                previous_segment[1] = previous_segment[2]
                previous_segment[0] = previous_segment[2]+1
            previous_segment = current_segment
        current_segment = segments[ordered_segments[-1]]
        current_segment[1] = len(words) # End at end of sentence.
        ordering = []
        for segment in segments:
            ordering.extend(range(segment[2], segment[0]-1, -1))
            ordering.extend(range(segment[2]+1, segment[1]))
        assert len(ordering) == len(words), pdb.set_trace()
        f.write('# ' + ' '.join([str(index) for index in ordering]) + '\n')
        for k, (w, t) in enumerate(instance):
            f.write('\t'.join([str(k), w, t]) + '\n')
        f.write('\n')

def main():
    filepath_train = sys.argv[1]
    filepath_dev = sys.argv[2]
    filepath_test = sys.argv[3]
    train_instances = list(read_dataset(filepath_train,
                                        maximum_sentence_length=-1))
    dev_instances = list(read_dataset(filepath_dev,
                                      maximum_sentence_length=-1))
    test_instances = list(read_dataset(filepath_test,
                                       maximum_sentence_length=-1))
    word_tag_dictionary, tags, word_entropies = create_word_tag_dictionary(
        train_instances, smoothing=1.)
    compute_easy_first_ordering(train_instances, word_entropies,
                                filepath_train + '.ordering',
                                easy_list_size=100)
    compute_easy_first_ordering(dev_instances, word_entropies,
                                filepath_dev + '.ordering',
                                easy_list_size=100)
    compute_easy_first_ordering(test_instances, word_entropies,
                                filepath_test + '.ordering',
                                easy_list_size=100)
    #pdb.set_trace()

if __name__ == "__main__":
    main()
