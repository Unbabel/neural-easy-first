# neural-easy-first baby steps

### Experiments on the WMT 2016 corpus

To get the data and the tools to handle it. You need the 
unbabel-quality-estimation repo 

    git clone https://github.com/Unbabel/unbabel-quality-estimation.git 

Right now it seesm that WMT2016 data on the web is faulty

    #cd unbabel-quality-estimation/wmt_corpora/
    #download_WMT_corpora.sh <your-data-folder>

You need to get the data from Ramon or Andre

This is an example script that trains system susing a given model. You can
modify this to include other models either in theano
or TF

    python train.py \
        -train-feat data/WMT2016/task2_en-de_training/train.basic_features_with_tags \
        -dev-feat data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags \
        -model-folder . \
        -model-type quetch \
        -embeddings_src data/WMT2016/data/embeddings/polyglot-de-wmt2016.txt  \
        -embeddings_trg data/WMT2016/data/embeddings/polyglot-en-wmt2016.txt  
