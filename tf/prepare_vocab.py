import argparse
import cPickle as pkl
from utils import load_embedding, load_vocabs
import os


def main(args):
    """
    extend the embeddings by training vocabulary
    note that the word2id mapping can contain more entries than the table,
    since several words can be mapped to the same embedding
    :param args:
    :return:
    """
    # load the embeddings
    src_embeddings = load_embedding(args.src_embeddings)
    tgt_embeddings = load_embedding(args.tgt_embeddings)

    # load the vocabulary (most frequent words)
    src_vocab, tgt_vocab = load_vocabs(args.train_src, args.train_tgt,
                                       args.train_data, args.src_limit,
                                       args.tgt_limit, args.freq_limit)

    # update the embeddings
    # add zero vectors for new words and for multiple alignment
    for src_word in src_vocab:
        if src_word not in src_embeddings.word2id.keys():
            src_embeddings.add_word(src_word)

    for tgt_word in tgt_vocab:
        if tgt_word not in tgt_embeddings.word2id.keys():
            tgt_embeddings.add_word(tgt_word)

    print "%d words were added to pre-trained src embeddings, %d tokens are multiple aligned words" \
          % (src_embeddings.added_words, src_embeddings.multiple_aligned_words)
    print "%d words were added to pre-trained tgt embeddings, %d tokens are multiple aligned words" \
          % (tgt_embeddings.added_words, tgt_embeddings.multiple_aligned_words)

    # dump embeddings
    train_name = os.path.basename(args.train_data)

    src_embeddings_file = "%s.%s.%d.min%d.extended.pkl" % (args.src_embeddings.split(".pkl")[0], train_name, args.src_limit, args.freq_limit)
    tgt_embeddings_file = "%s.%s.%d.min%d.extended.pkl" % (args.tgt_embeddings.split(".pkl")[0], train_name, args.tgt_limit, args.freq_limit)
    src_embeddings.store(src_embeddings_file)
    tgt_embeddings.store(tgt_embeddings_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare vocabulary and embeddings for nef.')
    parser.add_argument('train_src', type=str, help='the training source text')
    parser.add_argument('train_tgt', type=str, help='the training target text')
    parser.add_argument('train_features', type=str, help='training data set ("features with tags" format)')
    parser.add_argument('src_embeddings', type=str, help='the source embeddings')
    parser.add_argument('tgt_embeddings', type=str, help='the target embeddings')
    parser.add_argument('--src_limit', type=int, default=0, help='most frequent src words added from data to embedding')
    parser.add_argument('--tgt_limit', type=int, default=0, help='most frequent tgt words added from data to embedding')
    parser.add_argument('--freq_limit', type=int, default=0, help='only include words that occur more than this often')

    args = parser.parse_args()
    main(args)
