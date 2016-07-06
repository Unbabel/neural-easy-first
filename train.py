'''
Basic trainined with logging
'''
import codecs
import argparse
import logging 
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
                                                                 features='quetch',
                                                                 batch_size=200)
    
    nr_dev_batch, dev_batch = wmt2016data.get_batch_iterator('dev', 
                                                             features='quatch', 
                                                             batch_size=200)

    # LOAD MODEL
    if args.model_type == 'quetch':
        batch_update
        count_errors
    else:
        raise NotImplementedError()
    
    # TRAIN
    for n in range(args.nr_epoch):
        # Train
        cost = 0 
        for i in range(nr_train_batch):
            cost += batch_update(*train_batch(i)) 
        # Evaluation
        errors = 0
        for i in range(nr_dev_batch):
            errors += count_errors(*dev_batch(i)) 
        errors /= dev_data.nr_batch
    
