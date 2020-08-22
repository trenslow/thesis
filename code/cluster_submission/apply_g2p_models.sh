#$ -S /bin/bash
# make sure the proper python environment is active

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
GRAPHEME_DATA_DIR="$DIR/../../data/processed"
MODEL_OUT_DIR="$DIR/../../data/modeling"
# folder where built Sequitur resides
G2P_DIR="$DIR/../../g2p"

for lang in bul ces pol rus
do
    # Best performing model for Czech was trigram, rest were 5-gram
    if [ $lang = "ces" ]; then
        ngram="3"
    else
        ngram="5"
    fi
    # apply to train data, print stdout and stderr out to files
    echo "applying g2p model to ${lang} training data"
    python "$G2P_DIR/g2p.py" --model "$MODEL_OUT_DIR/g2p_model-$ngram.$lang" --apply "$GRAPHEME_DATA_DIR/training_articles.$lang"  --encoding utf-8 \
    > "$GRAPHEME_DATA_DIR/training_articles_g2p.$lang" &> "$GRAPHEME_DATA_DIR/training_articles_g2p.$lang.err"
    # apply to test data, print stdout and stderr out to files
    echo "applying g2p model to ${lang} test data"
    python "$G2P_DIR/g2p.py" --model "$MODEL_OUT_DIR/g2p_model-$ngram.$lang" --apply "$GRAPHEME_DATA_DIR/test_articles.$lang"  --encoding utf-8 \
    > "$GRAPHEME_DATA_DIR/test_articles_g2p.$lang" &> "$GRAPHEME_DATA_DIR/test_articles_g2p.$lang.err"
done