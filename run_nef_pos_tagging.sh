model_type=$1 #single_state
attention_type=$2 #sparsemax
temperature=$3 #1.
discount_factor=$4 #0.
l2_regularization=$5 #0.001
num_sketches=$6 #5
concatenate_last_layer=$7 #1
use_sketch_losses=$8
use_max_pooling=$9 # If 1, requires preattention_size == sketch_size
maximum_sentence_length=50 # 100
sum_hidden_states_and_sketches=${10} # If 1, requires sketch_size = 2*hidden_size.
share_attention_sketch_parameters=${11} # 0
num_epochs=20
num_pretraining_epochs=0
embedding_size=${12} #64
hidden_size=${13} #25 #30
preattention_size=${14} #50
sketch_size=${15} #50
context_size=${16}
noise_level=${17}
affix_length=${18}
affix_embedding_size=${19}
use_bilstm=${20}
dropout_probability=${21}
language=${22}

suffix=model-${model_type}_attention-${attention_type}_temp-${temperature}_\
disc-${discount_factor}_C-${l2_regularization}_sketches-${num_sketches}_\
cat-${concatenate_last_layer}_sum-${sum_hidden_states_and_sketches}_\
share-${share_attention_sketch_parameters}_\
skloss-${use_sketch_losses}_pool-${use_max_pooling}_\
bilstm-${use_bilstm}_pretrain-${num_pretraining_epochs}_\
emb-${embedding_size}_hid-${hidden_size}_pre-${preattention_size}_\
sk-${sketch_size}_ctx-${context_size}_drop-${dropout_probability}

echo $suffix

mkdir -p pos_tagging/logs/${language}
mkdir -p pos_tagging/sketches/${language}

python neftagger.py \
    --dynet-seed 42 \
    --dynet-mem 1024 \
    -num_epochs ${num_epochs} \
    -num_pretraining_epochs ${num_pretraining_epochs} \
    -train_file pos_tagging/data/${language}-normalized_train.conll.tagging \
    -dev_file pos_tagging/data/${language}-normalized_dev.conll.tagging \
    -test_file pos_tagging/data/${language}-normalized_test.conll.tagging \
    -embeddings_file pos_tagging/data/${language}.embeddings \
    -affix_length ${affix_length} \
    -model_type ${model_type} \
    -attention_type ${attention_type} \
    -concatenate_last_layer ${concatenate_last_layer} \
    -sum_hidden_states_and_sketches ${sum_hidden_states_and_sketches} \
    -share_attention_sketch_parameters ${share_attention_sketch_parameters} \
    -use_sketch_losses ${use_sketch_losses} \
    -use_max_pooling ${use_max_pooling} \
    -use_bilstm ${use_bilstm} \
    -temperature ${temperature} \
    -discount_factor ${discount_factor} \
    -num_sketches ${num_sketches} \
    -maximum_sentence_length ${maximum_sentence_length} \
    -l2_regularization ${l2_regularization} \
    -affix_embedding_size ${affix_embedding_size} \
    -embedding_size ${embedding_size} \
    -hidden_size ${hidden_size} \
    -preattention_size ${preattention_size} \
    -sketch_size ${sketch_size} \
    -context_size ${context_size} \
    -dropout_probability ${dropout_probability} \
    -sketch_file_dev pos_tagging/sketches/${language}/sketch_dev_${suffix}.txt \
    -sketch_file_test pos_tagging/sketches/${language}/sketch_test_${suffix}.txt \
    >& pos_tagging/logs/${language}/log_${suffix}.txt
