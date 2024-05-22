#!/bin/bash

# Impostazione dei parametri
SESSION_NAME="forward_screening_power"
DATA_DIR="/work/cozzoli_creanza/data/refined-set"
CASF_DIR="/work/cozzoli_creanza/data/CASF-2016"
MODEL_PATH="/work/cozzoli_creanza/output/trained_models/retrain.pth"
NUM_WORKERS=20
OUTPUT_PATH="/work/cozzoli_creanza/output/forward_screening_power/casf_input"

# Creazione di una nuova sessione tmux
tmux new -d -s "$SESSION_NAME"

# Attivazione dell'ambiente con micromamba
tmux send-keys -t "$SESSION_NAME" "micromamba activate genscore" Enter

# Esecuzione dello script Python
tmux send-keys -t "$SESSION_NAME" "python3 /work/cozzoli_creanza/GenScore/scripts/casf2016_forwardScreeningPower.py -d $DATA_DIR -csf $CASF_DIR -mp $MODEL_PATH --num_workers $NUM_WORKERS --outpath $OUTPUT_PATH" Enter

# Allega alla sessione tmux
echo "Per visualizzare lo stato di esecuzione esegui:"
echo tmux attach -t "$SESSION_NAME"
