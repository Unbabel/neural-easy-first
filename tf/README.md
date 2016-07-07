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


##2. Training a model##

  See all options
  
      python ef.py -h
    
  Train a model (and store in `models` directory)
  
      mkdir -p models
      python ef.py --train


##3. Testing a model##

  1. On batch data from stored model
  
    python ef.py

  2. Interactively
  
    python ef.py --interactive
