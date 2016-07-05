0. Tensorflow installation

TF has to be installed with specific URLs for each platform

    https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation

For Example Mac OS X, CPU only, Python 2.7, needs

    virtualenv tf
    source tf/bin/activate
    pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl

The rest can be installed with the usual

    pip install -r requirements.txt


1. Training a model

Options:
    `python ef.py -h

In tf directory, run
    `python ef.py --train`


2. Testing a model on batch data

    `python ef.py`


3. Testing the model interactively

    `python ef.py --interactive`