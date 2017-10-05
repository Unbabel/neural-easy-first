Before starting, install the Dynet version forked at https://github.com/andre-martins/dynet in your system.

To run easy-first taggers for POS tagging:

1. Put your text files in a folder pos_tagging/data.
2. Get your embeddings and run script filter_embeddings.py to get embeddings for your train, dev, and test files.
3. Run ```./run_nef_pos_tagging_all_01.sh en-ud constrained_softmax -1 0.2 &``` and look at the log file in pos_tagging/logs/english/

To run easy-first taggers for NER:

1. Put your text files in a folder ner/data.
2. Get your embeddings and run script filter_embeddings.py to get embeddings for your train, dev, and test files.
3. Run ```./run_nef_ner_all_01.sh english constrained_softmax 5 0.3 &``` and look at the log file in ner/logs/english/