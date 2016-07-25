warm_start_on_epoch=0
use_attention=$1 #1
attention_type=$2 #0
learning_rate=$3 #0.0003
regularization_constant=$4 #0.0003
cost_false_positives=$5 #0.2
cost_false_negatives=$6 #0.8

model_prefix=exp_bidirectional_attention-${use_attention}_type-${attention_type}_startepoch-${warm_start_on_epoch}_eta-${learning_rate}_lambda-${regularization_constant}_costs-${cost_false_positives}-${cost_false_negatives}

echo ${model_prefix}

mkdir -p ${model_prefix}

../attentive_quality_estimator train \
    ../data/training_small.txt ../data/dev.txt ../data/test.txt \
    ../data/word_vectors_jesus_en.sg.512.txt \
    ../data/word_vectors_jesus_es.sg.512.txt \
    ${cost_false_positives} \
    ${cost_false_negatives} \
    ${use_attention} \
    ${attention_type} \
    100 \
    ${warm_start_on_epoch} \
    20 \
    25 \
    ${learning_rate} \
    ${regularization_constant} \
    ${model_prefix}/  >& \
    log_${model_prefix}.txt

