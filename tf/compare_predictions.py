import argparse
import sys
import numpy as np
import random

def tags_to_binary(tagstring):
    map = {"OK": 0, "BAD": 1}
    tags = tagstring.split()
    return np.array([map[t] for t in tags])


def main(args):
    with open(args.pred1, "r") as f1, open(args.pred2, "r") as f2,\
        open(args.mt, "r") as mt, open(args.src, "r") as src, \
        open(args.pe, "r") as pe, open(args.tags, "r") as tagf:
        mtlines = mt.readlines()
        srclines = src.readlines()
        pelines = pe.readlines()
        taglines = tagf.readlines()
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        equals = []
        equal_and_correct = 0.
        better12 = []
        better21 = []
        if len(lines1) != len(lines2) != len(mtlines) != len(srclines) != len(pelines) != len(taglines):
            print "Files cannot be compared, line numbers do not agree"
            sys.exit(-1)
        else:
            i = 0
            for l1, l2, mt, src, pe, tags in zip(lines1, lines2, mtlines, srclines, pelines, taglines):
                a1 = np.fromstring(l1.strip()[1:-2], sep=",", dtype=int)
                a2 = np.fromstring(l2.strip()[1:-2], sep=",", dtype=int)
                equal = np.array_equal(a1, a2)
                true = tags_to_binary(tags.strip())
                #print i, a1, a2, np.sum(np.abs(np.subtract(a1, a2))), src.strip(), mt.strip(), pe.strip(), tags.strip(), true

                #print i, a1, a2, np.sum(np.abs(np.subtract(a1, a2))), src.strip(), mt.strip(), pe.strip(), true
                diff_a1 = np.sum(np.abs(np.subtract(a1, true)))
                diff_a2 = np.sum(np.abs(np.subtract(a2, true)))
                if diff_a1 < diff_a2:
                    better12.append((i, a1, a2, src.strip(), mt.strip(), pe.strip(), true))
                else:
                    better21.append((i, a1, a2, src.strip(), mt.strip(), pe.strip(), true))

                if equal:
                    if np.sum(a1-true) == 0:
                        equal_and_correct += 1
                    equals.append((i, a1, a2, src.strip(), mt.strip(), pe.strip(), true))
                i += 1
        print "Systems agree on %d predictions (of which %.2f%% are correct)" % (len(equals), equal_and_correct/len(equals)*100)
        print "System 1 is better in %d predictions" % len(better12)
        print "System 2 is better in %d predictions" % len(better21)

        for i in xrange(args.no_examples):
            i, a1, a2, src, mt, pe, true = random.choice(better12)
            print "System1 > System2"
            print "sentence %d" % i
            print "system 1", a1
            print "system 2", a2
            print "true    ", true
            print "mt", mt
            print "src", src
            print "pe", pe
            print

        for i in xrange(args.no_examples):
            i, a1, a2, src, mt, pe, true = random.choice(better21)
            print "System2 > System1"
            print "sentence %d" % i
            print "system 1", a1
            print "system 2", a2
            print "true    ", true
            print "mt", mt
            print "src", src
            print "pe", pe
            print



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two predictions of binary QE labels.')
    parser.add_argument('pred1', type=str, help='prediction 1')
    parser.add_argument('pred2', type=str, help='prediction 2')
    parser.add_argument('mt', type=str, help="machine translations")
    parser.add_argument('src', type=str, help="source sentences")
    parser.add_argument('pe', type=str, help="post editions")
    parser.add_argument('tags', type=str, help="QE tags")
    parser.add_argument('--no_examples', type=int, default=15, help="number of examples being reported")
    args = parser.parse_args()
    main(args)