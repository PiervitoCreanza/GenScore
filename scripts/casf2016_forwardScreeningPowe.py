import argparse
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
sys.path.append("/work/cozzoli_creanza/GenScore")
import pickle
from torch_geometric.loader import DataLoader
from GenScore.data.data import VSDataset
from GenScore.model.utils import run_an_eval_epoch
from GenScore.model.model import GenScore, GraphTransformer, GatedGCN
from alive_progress import alive_bar


def validate_encoder_argument(val):
    if not (val == "gt" or val == "gatedgcn"):
        raise argparse.ArgumentTypeError('Encoder must be either \'gt\' or \'gatedgcn\'')
    return val


def parse_user_input():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', required=True,
                   help='The path where the protein-ligand complexes are stored.')
    p.add_argument('--num_workers', type=int, default=10)
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


def scoring(prot, lig, kwargs):
    """
    prot: The input protein file ('.pdb')
    lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
    kwargs: other arguments related with model
    """
    try:
        data = VSDataset(ligs=lig,
                         prot=prot,
                         explicit_H=kwargs.useH,
                         use_chirality=kwargs.use_chirality,
                         parallel=kwargs.parallel)

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
                         dist_threhold=kwargs.dist_threhold
                         ).to(kwargs.device)

        checkpoint = th.load(kwargs.model_path, map_location=th.device(kwargs.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs.dist_threhold,
                                  device=kwargs.device)
        print("Successfully scored {} and {}".format(prot, lig))
        return data.ids, preds
    except:
        print("failed to score for {} and {}".format(prot, lig))
        return None, None


def score_compound(pdbid, ligids, args):
    print("Starting scoring of %s ..." % pdbid)
    ids_list = []
    scores_list = []
    for ligid in ligids:
        ids, scores = scoring(prot="%s/%s/%s_pocket.ligen.pdb" % (args.dir, pdbid, pdbid),
                              lig="%s/decoys_screening/%s/%s_%s.mol2" % (args.casf_path, pdbid, pdbid, ligid),
                              kwargs=args
                              )

        if ids is not None and scores is not None:
            ids_list.extend(ids)
            scores_list.extend(scores)
    print("Scoring of %s finished." % pdbid)
    return pdbid, [ids_list, scores_list]


def get_directories_inside_dir(dir):
    directories = []
    for element in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, element)):
            directories.append(element)

    return directories

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def main():
    args = parse_user_input()

    ligids_casfCoreSet = get_directories_inside_dir(os.path.join(args.casf_path, "coreset"))
    pdbids_casfDecoys = get_directories_inside_dir(os.path.join(args.casf_path, "decoys_screening"))
    pdbids_customSet = get_directories_inside_dir(args.dir)

    intersecting_pdbids = intersection(pdbids_casfDecoys, pdbids_customSet)

    if args.parallel:
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(score_compound)(pdbid, ligids_casfCoreSet, args) for pdbid in intersecting_pdbids)
    else:
        results = []
        with alive_bar(len(intersecting_pdbids)) as bar:
            for pdbid in intersecting_pdbids:
                results.append(score_compound(pdbid, ligids_casfCoreSet, args))
                bar()
    
    outdir = os.path.join(args.casf_path, args.outprefix)
    os.system("mkdir -p %s" % outdir)
    for res in results:
        pdbid = res[0]
        df = pd.DataFrame(zip(*res[1]), columns=["#code_ligand_num", "score"])
        df["#code_ligand_num"] = df["#code_ligand_num"].str.split("-").apply(lambda x: x[0])
        df.to_csv("%s/%s_score.dat" % (outdir, pdbid), index=False, sep="\t")

    with open("%s_screening.pkl" % args.outprefix, "wb") as dbFile:
        pickle.dump(results, dbFile)


if __name__ == '__main__':
    main()