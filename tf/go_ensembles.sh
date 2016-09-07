#!/bin/bash -e

# TRAINING
#for seed in 1234 2134 2314 2341 1324;do
#for seed in 2134 2314 2341 1324;do
#    model_dir="models/shuffle_seed${seed}/"
#    printf "$model_dir\n"
#    [ ! -d $model_dir ] && mkdir $model_dir
#    #python nef.py --train_file ../data/WMT2016/task2_en-de_training_shuffled/train.shuffle${index}.basic_features_with_tags \
#    python nef.py --train_file ../data/WMT2016/task2_en-de_training/train.features_with_tags \
#                  --dev_file ../data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags  \
#                  --bucket_random_seed $seed \
#                  --model_dir $model_dir \
#                  --train 
#
#done              

## TESTING 
#for seed in 1234 2134 2314 2341;do
#    model_dir="models/shuffle_seed${seed}/"
#    [ ! -d $model_dir ] && mkdir $model_dir
#
#    printf "\033[34mDEVEL $model_dir\033[0m\n"
#    python nef.py --test_file ../data/WMT2016/task2_en-de_dev/dev.basic_features_with_tags  \
#                  --model_dir $model_dir \
#                  --train False  \
#                  --save_pBAD True
#
#    printf "models/shuffle_seed${seed}/"
#    python ensemble/pred_from_pBAD.py $model_dir/dev.basic_features_with_tags.pBAD \
#                                      $model_dir/dev.basic_features_with_tags.pred
#
#    python ensemble/score.py \
#        -pred_tags_in $model_dir/dev.basic_features_with_tags.pred \
#        -gold_tags_in ../data/WMT2016/task2_en-de_dev/dev.tags  > $model_dir/dev.score
#
#done              

# Ensembling
# Majority voting
#[ ! -d "models/ensembles/" ] && mkdir models/ensembles/
python ./ensemble/majority_voting.py \
    -pred_tags_in models/shuffle_seed*/dev.basic_features_with_tags.pred \
    -pred_tags_out models/ensembles/shuffle_seed.pred

python ./ensemble/score.py \
     -pred_tags_in models/shuffle_seed*/dev.basic_features_with_tags.pred models/ensembles/shuffle_seed.pred \
     -gold_tags_in ../data/WMT2016/task2_en-de_dev/dev.tags  > models/ensembles/dev.majority_voting.score

printf "\033[34mMAJORITY VOTING\033[0m\n"
cat models/ensembles/dev.majority_voting.score

# Average probability 
#[ ! -d "models/ensembles/" ] && mkdir models/ensembles/
python ./ensemble/pBAD_product.py \
    -pBADs_in models/shuffle_seed*/dev.basic_features_with_tags.pBAD \
    -pBADs_out models/ensembles/shuffle_seed.pBAD \
    -pred_tags_out models/ensembles/shuffle_seed.average_prob.pred

python ./ensemble/score.py \
     -pred_tags_in models/shuffle_seed*/dev.basic_features_with_tags.pred models/ensembles/shuffle_seed.average_prob.pred \
     -gold_tags_in ../data/WMT2016/task2_en-de_dev/dev.tags > models/ensembles/dev.average_prob.score

printf "\033[34mAVERAGE PROBABILITY\033[0m\n"
cat models/ensembles/dev.average_prob.score
exit





#for index in $(seq 0 4);do
##    printf "Training models/shuffle${index}"
#    model_dir="models/shuffle${index}/"
##    [ ! -d $model_dir ] && mkdir $model_dir
##    python nef.py --train_file ../data/WMT2016/task2_en-de_training_shuffled/train.shuffle${index}.basic_features_with_tags \
##                  --test_file ../data/WMT2016/task2_en-de_test/test.corrected_full_parsed_features_with_tags \
##                  --model_dir $model_dir \
##                  --train False  \
##                  --save_pBAD True
#
#    python ensemble/pred_from_pBAD.py $model_dir/test.corrected_full_parsed_features_with_tags.pBAD \
#                                      $model_dir/test.corrected_full_parsed_features_with_tags.pred
#
#    python ensemble/score.py \
#        -pred_tags_in $model_dir/test.corrected_full_parsed_features_with_tags.pred \
#        -gold_tags_in ../data/WMT2016/task2_en-de_test/test.tags  
#
#done              
#
## SCORING
