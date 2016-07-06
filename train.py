'''
Basic trainined with logging
'''
import codecs
import argparse
import logging 

import theano

from ipdb import set_trace

# Local dependencies
import sys
sys.path.append('../unbabel-quality-estimation/wmt_corpora/tools/')
import wmt2016

if __name__ == '__main__':

    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(prog='Trains model')
    parser.add_argument('-nr-epoch', help='Number of training iterations', 
                        type=int, default=20)
    parser.add_argument('-batch-size', type=int, default=200)
    # 'data/WMT2016/task2_en-de_training/train.basic_features_with_tags'
    parser.add_argument('-train-feat', help='File with train features', 
                        type=str, required=True)
    parser.add_argument('-embeddings_src', help='File with embedings in text form', 
                        type=str, required=True)
    parser.add_argument('-embeddings_trg', help='File with embedings in text form', 
                        type=str, required=True)
    # 'data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags'
    parser.add_argument('-dev-feat', help='File with dev features', 
                        type=str, required=True)
    parser.add_argument('-model-folder', help='where model data will be stored', type=str, 
                        required=True)
    parser.add_argument('-model-type', help='type of model', type=str, 
                        required=True)
    # Parse
    args = parser.parse_args(sys.argv[1:])

    # LOGGER
    log_path = '%s/%s.log' % (args.model_folder, 'training')
    logging.basicConfig(level=logging.DEBUG,
                        filename=log_path,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    print "Log will be stored in %s" % log_path
    # Store config in log 
    logging.debug('Config')
    for arg, value in vars(args).items():
        default_value = parser.get_default(arg) 
        if default_value != value:
            logging.debug("\t%s = \033[34m%s\033[0m" % (arg, value))
        else:    
            logging.debug("\t%s = %s" % (arg, value))
 
    # LOAD WMT2016 FEATURES 
    wmt2016data = wmt2016.data(train_file=args.train_feat, dev_file=args.dev_feat)
    nr_train_batch, train_batch = wmt2016data.get_batch_iterator('train', 
        features='quetch', batch_size=200, dtype_input=theano.config.floatX, 
        dtype_target='int32')
    
    nr_dev_batch, dev_batch = wmt2016data.get_batch_iterator('dev', 
        features='quetch', batch_size=200, dtype_input=theano.config.floatX, 
        dtype_target='int32')

    # LOAD EMBEDDINGS
    E_src = wmt2016.load_embeddings(args.embeddings_src, wmt2016data.src_dict)
    E_trg = wmt2016.load_embeddings(args.embeddings_trg, wmt2016data.trg_dict) 

    # LOAD MODEL
    if args.model_type == 'theano-neural-easy-first':
        import theano_version.nef 
        model = theano_version.nef.NeuralEasyFirst(
            nr_classes=2, 
            emb_matrices=(E_src, E_trg))
    else:
        raise NotImplementedError("Unknown model %s" % args.model_type)
    
    # TRAIN
    for n in range(args.nr_epoch):
        # Train
        cost = 0 
        for i in range(nr_train_batch):
            cost += model.batch_update(*train_batch(i)) 
            if not i % 10:
                perc = (i+1)*1./nr_train_batch
                print "\r%d/%d (%2.1f %%)" % (i+1, nr_train_batch, perc),
        # Evaluation
        errors = 0
        nr_examples = 0
        for i in range(nr_dev_batch):
            hat_y = model.predict(*dev_batch(i)[:-1])
            y = dev_batch(i)[-1]
            errors += sum(y != hat_y)
            nr_examples += y.shape[0]
        errors /= nr_examples 

    # Info     
    logging.debug('Epoch %d/%d: Acc %2.2f %%' % (n+1, args.nr_epoch, 100*(1-errors)))
