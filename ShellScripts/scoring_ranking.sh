micromamba activate genscore

python3 /work/cozzoli_creanza/GenScore/scripts/scoring_ranking_custom.py -d /work/cozzoli_creanza/output/graphs/palermoSet_davideprots_plus_generalligs --file_prefix palermoSet -csf /work/cozzoli_creanza/data/CASF-2016 -mp /work/cozzoli_creanza/output/trained_models/palermoSet_davideprots_plus_generalligs.pth --num_workers 20 -o palermoSet_davideprots_plus_generalligs

micromamba activate casf

python scoring_power.py -c CoreSet.dat -s ./examples/palermoSet_davideprots_plus_generalligs.dat -p 'positive' -o 'X-Score' > /work/cozzoli_creanza/output/power_scoring/palermoSet_davideprots_plus_generalligs.out