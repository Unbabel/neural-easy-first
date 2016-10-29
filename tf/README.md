# Neural Easy-First Model in Tensorflow #

##1. Prerequisites##

  TF has to be installed with specific URLs for each platform

      https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation

  For Example Mac OS X, CPU only, Python 2.7, needs

      virtualenv tf
      source tf/bin/activate
      pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.10.0-py2-none-any.whl

  The rest can be installed with the usual

      pip install -r requirements.txt


##2. POS tagging ##

###2.1. Preparing the data ###
  
  The data format is tabular (one column for words, the second column for POS tags).

  Pre-trained embeddings as e.g. polyglot (pickle dumps) are merged with new vocabulary introduced by the task.

      mkdir -p pos_tagging/data
      mv en-ud-normalized_train.conll.tagging en-ud-normalized_dev.conll.tagging en-ud-normalized_test.conll.tagging pos_tagging/data
      mkdir -p pos_tagging/data/embeddings
      mv polyglot-en.pkl pos_tagging/data/embeddings

###2.2. Training a model###

  Baseline (BILSTM) and Neural-Easy-First models are both implemented in the same code base.
  
  To run them with according hyper-parameter settings, see all options:
  
      python nef.py -h

  Train a BILSTM model (and store in `pos_tagging/models` directory):
  
      mkdir -p pos_tagging/models
      python nef.py --num_sketches=0

  Train a Neural-Easy-First model:

      python nef.py

###2.3 Testing a model###

  On batch data from stored model:

     python nef.py --train False

  Interactively:

     python nef.py --interactive


##3. Quality estimation ##

###3.1. Preparing the data ###
  
  The data has to be in the format as distributed at WMT2016.

  Pre-trained embeddings as e.g. polyglot (pickle dumps) are merged with new vocabulary introduced by the task.

      mkdir -p quality_estimation/data
      mkdir -p quality_estimation/data/WMT2016/task2_en-de_training
      mkdir -p quality_estimation/data/WMT2016/task2_en-de_dev
      mkdir -p quality_estimation/data/WMT2016/task2_en-de_test
      mv train.basic_features_with_tags quality_estimation/data/WMT2016/task2_en-de_training
      mv dev.basic_features_with_tags quality_estimation/data/WMT2016/task2_en-de_dev
      mv test.basic_features_with_tags quality_estimation/data/WMT2016/task2_en-de_test
      mkdir -p quality_estimation/data/embeddings
      mv polyglot-en.pkl quality_estimation/data/embeddings
      mv polyglot-de.pkl quality_estimation/data/embeddings

###3.2. Training a model###

  Baseline (QUETCH-BILSTM) and Neural-Easy-First models are both implemented in the same code base.
  
  To run them with according hyper-parameter settings, see all options:
  
      python nef.py -h

  Train a QUETCH-BILSTM model (and store in `models` directory):
  
      mkdir -p models
      python nef.py \
          --task=quality_estimation \
          --num_sketches=0 \
          --embeddings=quality_estimation/data/embeddings/polyglot-de.pkl,quality_estimation/data/embeddings/polyglot-en.pkl \
          --embedding_sizes=64,64 \
          --model_dir=quality_estimation/models \
          --training_file=quality_estimation/data/WMT2016/task2_en-de_training/train.basic_features_with_tags \
          --dev_file=quality_estimation/data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags \
          --test_file=quality_estimation/data/WMT2016/task2_en-de_test/test.basic_features_with_tags \
          --train=True

  Note that the arguments --embeddings and --embedding_sizes have two values separated by a comma.
  The first value corresponds to the target embeddings and the second one to the source embeddings.

  Train a Neural-Easy-First model:
  
      python nef.py \
          --task=quality_estimation \
          --embeddings=quality_estimation/data/embeddings/polyglot-de.pkl,quality_estimation/data/embeddings/polyglot-en.pkl \
          --embedding_sizes=64,64 \
          --model_dir=quality_estimation/models \
          --training_file=quality_estimation/data/WMT2016/task2_en-de_training/train.basic_features_with_tags \
          --dev_file=quality_estimation/data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags \
          --test_file=quality_estimation/data/WMT2016/task2_en-de_test/test.basic_features_with_tags \
          --train=True

###3.3 Testing a model###

  1. On batch data from stored model:
  
      python nef.py \
          --task=quality_estimation \
          --embeddings=quality_estimation/data/embeddings/polyglot-de.pkl,quality_estimation/data/embeddings/polyglot-en.pkl \
          --embedding_sizes=64,64 \
          --model_dir=quality_estimation/models \
          --training_file=quality_estimation/data/WMT2016/task2_en-de_training/train.basic_features_with_tags \
          --dev_file=quality_estimation/data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags \
          --test_file=quality_estimation/data/WMT2016/task2_en-de_test/test.basic_features_with_tags \
          --train=False

  2. Interactively:
  
      python nef.py \
          --task=quality_estimation \
          --embeddings=quality_estimation/data/embeddings/polyglot-de.pkl,quality_estimation/data/embeddings/polyglot-en.pkl \
          --embedding_sizes=64,64 \
          --model_dir=quality_estimation/models \
          --training_file=quality_estimation/data/WMT2016/task2_en-de_training/train.basic_features_with_tags \
          --dev_file=quality_estimation/data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags \
          --test_file=quality_estimation/data/WMT2016/task2_en-de_test/test.basic_features_with_tags \
          --interactive
