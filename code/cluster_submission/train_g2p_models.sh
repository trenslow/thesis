#$ -S /bin/bash
# make sure the proper python environment is active

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
TRAIN_DATA_DIR="$DIR/../../data/processed"
MODEL_OUT_DIR="$DIR/../../data/modeling"
# folder where built Sequitur resides
G2P_DIR="$DIR/../../g2p"

echo $DIR
echo $TRAIN_DATA_DIR
echo $MODEL_OUT_DIR

for lang in bul ces pol rus
do
    for ngram in 1 2 3 4 5
    do
        last_ngram=$((ngram-1))
        if [ $ngram = 1 ]; then
            python "$G2P_DIR/g2p.py" --train "$TRAIN_DATA_DIR/g2p_train_data.$lang" --devel 5% --write-model "$MODEL_OUT_DIR/g2p_model-$ngram.$lang" --encoding utf-8 
        else
            python "$G2P_DIR/g2p.py" --model "$MODEL_OUT_DIR/g2p_model-$last_ngram.$lang" --ramp-up --train "$TRAIN_DATA_DIR/g2p_train_data.$lang" --devel 5% --write-model "$MODEL_OUT_DIR/g2p_model-$ngram.$lang" --encoding utf-8
        fi
    done
done