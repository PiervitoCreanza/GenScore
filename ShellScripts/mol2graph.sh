#!/bin/bash

# Creazione di una nuova sessione tmux
tmux new -d -s "mol2graph"

# Attivazione dell'ambiente con micromamba
tmux send-keys -t "mol2graph" "micromamba activate genscore" Enter

# Esecuzione dello script Python
tmux send-keys -t "mol2graph" "python3 /work/cozzoli_creanza/GenScore/GenScore/feats/mol2graph_rdmda_res.py -p -uschi -d /work/cozzoli_creanza/refined-set -r /work/cozzoli_creanza/input/no_dup_INDEX_comprehensive_data.csv -fs /work/cozzoli_creanza/input/Palermo_training-set.csv -o /work/cozzoli_creanza/output/graphs/palermoSet_davideprots_plus_generalligs/palermoSet -rc /work/cozzoli_creanza/input/INDEX_refined_data_2020_list_sorted.txt -dg /work/cozzoli_creanza/input/missing-prots" Enter

# Allega alla sessione tmux
echo "Per visualizzare lo stato di esecuzione esegui:"
echo tmux attach -t "mol2graph"