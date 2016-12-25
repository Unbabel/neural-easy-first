model_type=$1 #single_state
attention_type=$2 #sparsemax
temperature=$3 #1.
discount_factor=$4 #0.
l2_regularization=$5 #0.001
num_sketches=$6 #5
concatenate_last_layer=$7 #1
embedding_size=64
hidden_size=20
context_size=1

suffix=model-${model_type}_attention-${attention_type}_temp-${temperature}_\
disc-${discount_factor}_C-${l2_regularization}_sketches-${num_sketches}_\
cat-${concatenate_last_layer}_emb-${embedding_size}_hid-${hidden_size}_\
ctx-${context_size}

echo $suffix

python neftagger.py \
    --dynet-seed 42 \
    -num_epochs 20 \
    -train_file ../tf/pos_tagging/data/en-ud-normalized_train.conll.tagging \
    -dev_file ../tf/pos_tagging/data/en-ud-normalized_dev.conll.tagging \
    -test_file ../tf/pos_tagging/data/en-ud-normalized_test.conll.tagging \
    -model_type ${model_type} \
    -attention_type ${attention_type} \
    -concatenate_last_layer ${concatenate_last_layer} \
    -temperature ${temperature} \
    -discount_factor ${discount_factor} \
    -num_sketches ${num_sketches} \
    -l2_regularization ${l2_regularization} \
    -embedding_size ${embedding_size} \
    -hidden_size ${hidden_size} \
    -context_size ${context_size} \
    -sketch_file sketches/sketch_${suffix}.txt \
    >& logs/log_${suffix}.txt



