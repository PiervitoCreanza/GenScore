import numpy as np
import torch as th
import pandas as pd
import os, sys
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# This script prepares the data to perform the scoring power and ranking power tests.
# The scoring power measursed the ability of a scoring function
# to accurately predict the binding affinity of a protein-ligand complex.
# The ranking power instead evaluates the ability to rank different complexes
# based on their binding affinities.
sys.path.append("/work/cozzoli_creanza/GenScore")
from torch_geometric.loader import DataLoader
from GenScore.data.data import PDBbindDataset
from GenScore.model.model import GenScore, GraphTransformer, GatedGCN
from GenScore.model.utils import run_an_eval_epoch
import torch.multiprocessing
import argparse
torch.multiprocessing.set_sharing_strategy('file_system')


def validate_encoder_argument(val):
    if not (val == "gt" or val == "gatedgcn"):
        raise argparse.ArgumentTypeError('Encoder must be either \'gt\' or \'gatedgcn\'')
    return val


def parse_user_input():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', required=True,
                   help='The path where the protein-ligand complexes are stored.')
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--file_prefix', required=True)
    p.add_argument('-csf', '--casf_path', required=True,
                   help='The path of the casf folder')
    p.add_argument('-mp', '--model_path', required=True,
                   help='The path of the model')
    p.add_argument('-c', '--cutoff', type=float, default=10.0,
                   help='the cutoff to determine the pocket')
    p.add_argument('-o', '--outprefix', default="out",
                   help='The output directory path.')
    p.add_argument('-usH', '--useH', default=False, action="store_true",
                   help='whether to use the explicit H atoms.')
    p.add_argument('-uschi', '--use_chirality', default=True, action="store_true",
                   help='whether to use chirality.')
    p.add_argument('-e', '--encoder', default="gt", type=validate_encoder_argument,
                   help='The reference file to query the label of the complex.')
    p.add_argument('-p', '--parallel', default=False, action="store_true",
                   help='whether to obtain the graphs in parallel (When the dataset is too large, '
                        'it may be out of memory when conducting the parallel mode).')
    args = p.parse_args()

    # Set costant arguments.
    args.batch_size = 64
    args.dist_threhold = 5.
    args.device = 'cuda' if th.cuda.is_available() else 'cpu'
    args.seeds = 126
    args.num_node_featsp = 41
    args.num_node_featsl = 41
    args.num_edge_featsp = 5
    args.num_edge_featsl = 10
    args.hidden_dim0 = 128
    args.hidden_dim = 128
    args.n_gaussians = 10
    args.dropout_rate = 0.15
    return args


def scoring(ids, prots, ligs, modpath,kwargs):
    """
    prot: The input protein file ('.pdb')
    lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
    modpath: The path to store the pre-trained model
    gen_pocket: whether to generate the pocket from the protein file.
    reflig: The reference ligand to determine the pocket.
    cutoff: The distance within the reference ligand to determine the pocket.
    explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
    use_chirality: whether to adopt the information of chirality to represent the molecules.	
    parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
    kwargs: other arguments related with model
    """
    # try:
    data = PDBbindDataset(ids=ids, prots=prots, ligs=ligs)

    test_loader = DataLoader(dataset=data,
                             batch_size=kwargs.batch_size,
                             shuffle=False,
                             num_workers=kwargs.num_workers)

    if kwargs.encoder == "gt":
        ligmodel = GraphTransformer(in_channels=kwargs.num_node_featsl,
                                    edge_features=kwargs.num_edge_featsl,
                                    num_hidden_channels=kwargs.hidden_dim0,
                                    activ_fn=th.nn.SiLU(),
                                    transformer_residual=True,
                                    num_attention_heads=4,
                                    norm_to_apply='batch',
                                    dropout_rate=0.15,
                                    num_layers=6
                                    )

        protmodel = GraphTransformer(in_channels=kwargs.num_node_featsp,
                                     edge_features=kwargs.num_edge_featsp,
                                     num_hidden_channels=kwargs.hidden_dim0,
                                     activ_fn=th.nn.SiLU(),
                                     transformer_residual=True,
                                     num_attention_heads=4,
                                     norm_to_apply='batch',
                                     dropout_rate=0.15,
                                     num_layers=6
                                     )
    elif kwargs.encoder == "gatedgcn":
        ligmodel = GatedGCN(in_channels=kwargs.num_node_featsl,
                            edge_features=kwargs.num_edge_featsl,
                            num_hidden_channels=kwargs.hidden_dim0,
                            residual=True,
                            dropout_rate=0.15,
                            equivstable_pe=False,
                            num_layers=6
                            )

        protmodel = GatedGCN(in_channels=kwargs.num_node_featsp,
                             edge_features=kwargs.num_edge_featsp,
                             num_hidden_channels=kwargs.hidden_dim0,
                             residual=True,
                             dropout_rate=0.15,
                             equivstable_pe=False,
                             num_layers=6
                             )
    else:
        raise ValueError("encoder should be \"gt\" or \"gatedgcn\"!")

    model = GenScore(ligmodel, protmodel,
                     in_channels=kwargs.hidden_dim0,
                     hidden_dim=kwargs.hidden_dim,
                     n_gaussians=kwargs.n_gaussians,
                     dropout_rate=kwargs.dropout_rate,
                     dist_threhold=kwargs.dist_threhold).to(kwargs.device)

    checkpoint = th.load(modpath, map_location=th.device(kwargs.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs.dist_threhold,
                              device=kwargs.device)
    return data.pdbids, preds

def obtain_metrics(df):
    # Calculate the Pearson correlation coefficient
    regr = linear_model.LinearRegression()
    regr.fit(df.score.values.reshape(-1, 1), df.logKa.values.reshape(-1, 1))
    preds = regr.predict(df.score.values.reshape(-1, 1))
    rp = pearsonr(df.logKa, df.score)[0]
    # rp = df[["logKa","score"]].corr().iloc[0,1]
    mse = mean_squared_error(df.logKa, preds)
    num = df.shape[0]
    sd = np.sqrt((mse * num) / (num - 1))
    # return rp, sd, num
    print("The regression equation: logKa = %.2f + %.2f * Score" % (float(regr.coef_), float(regr.intercept_)))
    print("Number of favorable sample (N): %d" % num)
    print("Pearson correlation coefficient (R): %.3f" % rp)
    print("Standard deviation in fitting (SD): %.2f" % sd)

def main():
    args = parse_user_input()
    prots = '%s/%s_prot.pt' % (args.dir, args.file_prefix)
    ligs = '%s/%s_lig.pt' % (args.dir, args.file_prefix)
    ids = '%s/%s_ids.npy' % (args.dir, args.file_prefix)
    
    _, preds = scoring(ids, prots, ligs, args.model_path, args)
    CoreSetPath = os.path.join(args.casf_path, "power_scoring/CoreSet.dat")
    df = pd.read_csv(CoreSetPath, sep='[,,\t, ]+', header=0, engine='python')
    df_score = pd.DataFrame(zip(np.load(ids)[0], preds), columns=["#code", "score"])
    testdf = pd.merge(df, df_score, on='#code')
    outFileName = '%s.dat' % args.outprefix
    outPath = os.path.join(args.casf_path, "power_scoring/examples", outFileName)
    testdf[["#code", "score"]].to_csv(outPath, index=False, sep=" ")

if __name__ == "__main__":
    main()



'''
-d
/work/cozzoli_creanza/output/graphs/palermoSet_davideprots_plus_generalligs
--file_prefix
palermoSet
-csf
/work/cozzoli_creanza/data/CASF-2016
-mp
/work/cozzoli_creanza/output/trained_models/palermoSet_davideprots_plus_generalligs.pth
--num_workers
20
-o
palermoSet_davideprots_plus_generalligs

-d
/work/cozzoli_creanza/data/rtmscore_s
--file_prefix
v2020_casf
-csf
/work/cozzoli_creanza/data/CASF-2016
-mp
/work/cozzoli_creanza/output/trained_models/retrain.pth
--num_workers
20
-o
retrain_on_zenodo_data_scoring_ranking
'''


