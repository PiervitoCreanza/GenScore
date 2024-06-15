# Machine Learning for drug-target interaction estimation in Drug Discovery: from replication to data integration analysis
This project focuses on the analysis and the retraining of a convolutional neural network,
GenScore, designed to calculate the affinity degree (score) between a protein and a ligand. The main
objective is to adapt this network to work with a new set of data produced by Politecnico di Milano.
This new data has a distinctive feature: the pockets of the proteins have a smaller cutout compared to
what was originally planned for the network. On one hand, this modification makes the data more
manageable and less computationally burdensome; on the other hand, it becomes necessary to
investigate how this model reacts to a stimulus so different from what was originally conceived for it.
This requires a very patient and particularly meticulous examination of the new results produced, in
order to understand their accuracy in terms of precision and efficiency. Through this retraining, we
therefore hope to improve the ability of the network to work with different sizes of cutouts of the
protein pockets, thus increasing its versatility and applicability in various scientific contexts. We hope
that the results of this project, in our small contribution, may have significant implications for future
research in the field of bioinformatics and computational chemistry.

_Read the full article here:_ [Machine Learning for drug-target interaction estimation in Drug Discovery: from replication to data integration analysis](https://polimi365-my.sharepoint.com/:b:/g/personal/10794727_polimi_it/EYnD7tjC1WdHrdeqHlCMQBcBPoxH7huPWohWtePMqVRNMQ?e=Hnh30a)

---
# GenScore (Original)
GenScore is a generalized protein-ligand scoring framework extended from RTMScore, and it exhibits balanced scoring, ranking, docking and screening powers on multiple datasets.

<div align=center>
<img src="https://github.com/sc8668/GenScore/blob/main/zzz-3.jpg" width="600px" height="300px">
</div> 



### Requirements
mdanalysis==2.0.0    
pandas==1.0.3    
prody==2.1.0    
python==3.8.11    
pytorch==1.11.0    
torch-geometric==2.0.3    
torch-scatter==2.0.9     
rdkit==2021.03.5    
openbabel==3.1.0    
scikit-learn==0.24.2    
scipy==1.6.2    
seaborn==0.11.2    
numpy==1.20.3    
pandas==1.3.2    
matplotlib==3.4.3   
joblib==1.0.1    

```
conda create --prefix xxx --file ./requirements_conda.txt      
pip install -r ./requirements_pip.txt
```
### Datasets
[PDBbind](http://www.pdbbind.org.cn)       
[CASF-2016](http://www.pdbbind.org.cn)    
[docking poses for DEKOIS2.0 and DUD-E](https://www.zenodo.org/record/6859325)   
[CSAR NRC-HiQ benchmark](http://www.csardock.org/)    
[Merck FEP benchmark](https://github.com/MCompChem/fep-benchmark)   
[PDBbind-CrossDocked-Core](https://www.zenodo.org/record/5525936)         

### Examples for using the trained model for prediction
```
cd example
```
___# input is protein (need to extract the pocket first)___
```
python genscore.py -p ./1qkt_p.pdb -l ./1qkt_decoys.sdf -rl ./1qkt_l.sdf -gen_pocket -c 10.0 -e gt -m ../trained_models/GT_0.0_1.pth
```
___# input is pocket___
```
python genscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -e gatedgcn -m ../trained_models/GatedGCN_0.5_1.pth
```
___# calculate the atom contributions of the score___
```
python genscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -e gatedgcn -ac -m ../trained_models/GatedGCN_ft_1.0_1.pth
```
___# calculate the residue contributions of the score___
```
python genscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -e gatedgcn -rc -m ../trained_models/GatedGCN_ft_1.0_1.pth
```

