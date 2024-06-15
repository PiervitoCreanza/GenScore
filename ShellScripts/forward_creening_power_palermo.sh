#!/bin/bash

# Impostazione dei parametri
SESSION_NAME="forward_screening_power"
REFINED_SET_DATA_DIR="/work/cozzoli_creanza/data/refined-set"
GENERAL_SET_DATA_DIR="/work/cozzoli_creanza/input/missing-prots"
CASF_DIR="/work/cozzoli_creanza/data/CASF-2016"
MODEL_PATH="/work/cozzoli_creanza/output/trained_models/palermoSet_davideprots_plus_generalligs.pth"
REFINED_SET_IDS="/work/cozzoli_creanza/input/INDEX_refined_data_2020_list_sorted.txt"
NUM_WORKERS=20
OUTPUT_PATH="/work/cozzoli_creanza/output/forward_screening_power/palermoSet_davideprots_plus_generalligs/casf_input"

# Creazione di una nuova sessione tmux
tmux new -d -s "$SESSION_NAME"

# Attivazione dell'ambiente con micromamba
tmux send-keys -t "$SESSION_NAME" "micromamba activate genscore" Enter

# Esecuzione dello script Python
echo "$SESSION_NAME" "python3 /work/cozzoli_creanza/GenScore/scripts/casf2016_forwardScreeningPower_palermo.py -d $REFINED_SET_DATA_DIR -dg $GENERAL_SET_DATA_DIR -rc $REFINED_SET_IDS  -csf $CASF_DIR -mp $MODEL_PATH --num_workers $NUM_WORKERS --outpath $OUTPUT_PATH" Enter

# Allega alla sessione tmux
echo "Per visualizzare lo stato di esecuzione esegui:"
echo tmux attach -t "$SESSION_NAME"
