python neftagger.py \
    -train_file ~/workspace/unbabel-x/neural-easy-first/tf/pos_tagging/data/en-ud-normalized_train.conll.tagging \
    -dev_file ~/workspace/unbabel-x/neural-easy-first/tf/pos_tagging/data/en-ud-normalized_dev.conll.tagging \
    -test_file ~/workspace/unbabel-x/neural-easy-first/tf/pos_tagging/data/en-ud-normalized_test.conll.tagging \
    -concatenate_last_layer 1
