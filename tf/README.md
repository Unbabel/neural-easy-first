# Neural Easy-First Model in Tensorflow #

##1. Prerequisites##

  TF has to be installed with specific URLs for each platform

      https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation

  For Example Mac OS X, CPU only, Python 2.7, needs

      virtualenv tf
      source tf/bin/activate
      pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl

  The rest can be installed with the usual

      pip install -r requirements.txt


##2. Preparing the data##
  
  The data has to be in the format as distributed at WMT2016.
  Pre-trained embeddings as e.g. polyglot (pickle dumps) are merged with new vocabulary introduced by the task.
  To construct the vocabulary and the initial lookup table, use the following script:
  
      python prepare_vocab.py <path to training src> <path to training tgt> <path to training feature file>  <path to src embeddings> <path to tgt embeddings> --freq_limit <freq_limit> --tgt_limit <tgt_limit> --src_limit <src_limit>
  
  This creates a new embedding dump including the `src_limit` and `tgt_limit` most frequent new words that occur more than `freq_limit` times on either source or target size.

##3. Training a model##

  Baseline (QUETCH) and Neural-Easy-First models are both implemented in the same code base.
  
  To run them with according hyper-parameter settings, see all options:
  
      python nef.py -h

  Train a model (and store in `models` directory)
  
      mkdir -p models
      python nef.py --train True


##4. Testing a model##

  1. On batch data from stored model
  
     python nef.py --train False

  2. Interactively
  
     python nef.py --interactive
