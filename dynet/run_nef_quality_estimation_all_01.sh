language=$1 #en-de_wmt16
attention_type=$2
model_type=single_state
temperature=1.
discount_factor=0.
num_sketches=$3 #-1
use_sketch_losses=0
embedding_size=64
hidden_size=50
preattention_size=50
context_size=2
noise_level=0.
affix_length=0 #4
affix_embedding_size=0 #50
use_bilstm=1

concatenate_last_layer=0
use_max_pooling=0
sum_hidden_states_and_sketches=0
share_attention_sketch_parameters=0
sketch_size=50

for bad_weight in 5 3
do
    for concatenate_last_layer in 0 1
    do
        for dropout_probability in 0 0.1 0.2 0.3 0.4 0.5
        do
            for l2_regularization in 0
            do
                ./run_nef_quality_estimation.sh \
                    ${model_type} \
                    ${attention_type} \
                    ${temperature} \
                    ${discount_factor} \
                    ${l2_regularization} \
                    ${num_sketches} \
                    ${concatenate_last_layer} \
                    ${use_sketch_losses} \
                    ${use_max_pooling} \
                    ${sum_hidden_states_and_sketches} \
                    ${share_attention_sketch_parameters} \
                    ${embedding_size} \
                    ${hidden_size} \
                    ${preattention_size} \
                    ${sketch_size} \
                    ${context_size} \
                    ${noise_level} \
                    ${affix_length} \
                    ${affix_embedding_size} \
                    ${use_bilstm} \
                    ${dropout_probability} \
                    ${bad_weight} \
                    ${language}
            done
        done
    done
done


