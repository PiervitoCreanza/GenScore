# Impostazione dei parametri
SOURCEPATH="/work/cozzoli_creanza/output/forward_screening_power/palermoSet_davideprots_plus_generalligs/casf_input"
OUTPUTNAME="palermoSet_davideprots_plus_generalligs_forwardScreeningPower"


# Attivazione dell'ambiente con micromamba
micromamba activate casf

# Esecuzione dello script Python
python /work/cozzoli_creanza/data/CASF-2016/power_screening/forward_screening_power.py -c CoreSet.dat -s $SOURCEPATH -t ./TargetInfo.dat -p 'positive' -o 'genscore_eval' > /work/cozzoli_creanza/output/forward_screening_power/casf_output/$OUTPUTNAME.out
