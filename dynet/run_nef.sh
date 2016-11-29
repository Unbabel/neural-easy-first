python neftagger.py \
    -train_file ../tf/pos_tagging/data/en-ud-normalized_train.conll.tagging \
    -dev_file ../tf/pos_tagging/data/en-ud-normalized_dev.conll.tagging \
    -test_file ../tf/pos_tagging/data/en-ud-normalized_test.conll.tagging \
    -concatenate_last_layer 1

#    -num_sketches 0 \
#    -embedding_size 200 \
#    -hidden_size 50

