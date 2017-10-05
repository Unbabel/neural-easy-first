

python nef.py \
    --src_embeddings ../data/WMT2016/embeddings/polyglot-en.train_full.features.0.min20.extended.pkl \
    --tgt_embeddings ../data/WMT2016/embeddings/polyglot-de.train_full.features.0.min20.extended.pkl \
    --notrack_sketches \
    --J 10 \
    --attention_temperature 1.0 \
    --attention_discount_factor 0.0 \
    --noconcat \
    --keep_prob 0.5
    #--lstm_units 150 \

#    --attention_temperature 0.2 \
#    --attention_discount_factor 5.0 \
