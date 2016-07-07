import numpy as np
import random

# generate dummy data for sequence tagging problems


def random_data(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are unrelated to xs and random
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x = np.floor(np.random.rand(number_of_instances, instance_length)*vocab_size)
        y = np.floor(np.random.rand(number_of_instances, instance_length)*number_of_labels)
        xs.append(x)
        ys.append(np.array(y))
    return xs, ys


def random_deterministic_data(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are dependent on xs
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x = np.floor(np.random.rand(number_of_instances, instance_length)*vocab_size)
        f = float(number_of_labels)/vocab_size
        y = [np.floor(x_i*f) for x_i in x]
        xs.append(x)
        ys.append(np.array(y))
    return xs, ys

def random_interdependent_data(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are dependent on xs and on previous xs (bi-gram)
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances: # for each dataset
        x = np.floor(np.random.rand(number_of_instances, instance_length)*vocab_size)
        probs = np.random.randn(vocab_size, number_of_labels, number_of_labels)
        y = [[-1]*instance_length for x_i in x]
        for i in range(number_of_instances):
            label = random.randint(0, number_of_labels-1)
            label_prev = label
            y[i][0] = label
            for j in range(1, instance_length):
                x_i_j = int(x[i][j])
                # fixed label transitions
                # alternatively use probabilistic transitions: np.random.choice(range(labels), p=probs)
                label = np.argmax(probs[x_i_j][label_prev], 0)
                y[i][j] = label
                label_prev = label
        xs.append(x)
        ys.append(np.array(y))
    return xs, ys


if __name__ == "__main__":
    print random_data(3, 4, 10, 3)
    print random_deterministic_data(3, 4, 10, 3)
    print random_interdependent_data(2, 5, 4, 3)