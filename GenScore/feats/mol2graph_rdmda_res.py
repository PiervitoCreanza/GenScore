import alive_progress
import pandas as pd
import numpy as np
from rdkit import Chem
import torch as th
import re, os
from itertools import permutations
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from joblib import Parallel, delayed
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances

from alive_progress import alive_bar
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import math

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24

n = 0


def prot_to_graph(prot, cutoff):
    """obtain the residue graphs"""
    u = mda.Universe(prot)
    # Add nodes
    num_residues = len(u.residues)

    res_feats = np.array([calc_res_features(res) for res in u.residues])
    edgeids, distm = obatin_edge(u, cutoff)
    src_list, dst_list = zip(*edgeids)

    ca_pos = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    center_pos = th.tensor(u.atoms.center_of_mass(compound='residues'))
    dis_matx_ca = distance_matrix(ca_pos, ca_pos)
    cadist = th.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    dis_matx_center = distance_matrix(center_pos, center_pos)
    cedist = th.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1
    edge_connect = th.tensor(np.array([check_connect(u, x, y) for x, y in edgeids]))
    edge_feats = th.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), th.tensor(distm)], dim=1)

    # res_max_natoms = max([len(res.atoms) for res in u.residues])
    res_coods = th.tensor(np.array(
        [np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS - len(res.atoms), 3), np.nan)], axis=0) for res in
         u.residues]))
    g = Data(x=th.tensor(res_feats, dtype=th.float),
             edge_index=th.tensor([src_list, dst_list]),
             pos=res_coods,
             edge_attr=th.tensor(np.array(edge_feats), dtype=th.float))

    # g.ndata.pop("ca_pos")
    # g.ndata.pop("center_pos")
    # g.ndata["pos"] = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
    return g


def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  ##some residues loss the CA atoms
            return res.atoms.positions.mean(axis=0)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
    try:
        # xx = res.atoms.select_atoms("not name H*")
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]


def calc_res_features(res):
    result = np.array(one_of_k_encoding_unk(obtain_resname(res),
                                            ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                             'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                             'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                             'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
                      obtain_self_dist(res) +  # 5
                      obtain_dihediral_angles(res)  # 4
                      )
    if result.size != 41:
        raise ValueError(f"Unexpected number of features: Got {result.size}")
    return result


def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
    cutoff = 10.0 if cutoff is None else cutoff  # Default value not set if None is paased
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0


def calc_dist(res1, res2):
    # xx1 = res1.atoms.select_atoms('not name H*')
    # xx2 = res2.atoms.select_atoms('not name H*')
    # dist_array = distances.distance_array(xx1.positions,xx2.positions)
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


# return dist_array.max()*0.1, dist_array.min()*0.1


def calc_atom_features(atom, explicit_H):
    if explicit_H is None: explicit_H = False
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    results = (one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'C', 'N', 'O', 'S', 'F', 'P', 'Cl',
            'Br', 'I', 'B', 'Si', 'Fe', 'Zn',
            'Cu', 'Mn', 'Mo', 'other'
        ]
    )  # 17
               +
               one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])  # +7 = 24
               +
               [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]  # +2 = 26
               +
               one_of_k_encoding_unk(atom.GetHybridization(), [
                   Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                   Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                   Chem.rdchem.HybridizationType.SP3D2, 'other'])  # + 6 = 32
               +
               [atom.GetIsAromatic()])  # + 1 = 33
    # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # + 5 = 38
    return np.array(results)


def calc_bond_features(bond, use_chirality):
    if use_chirality is None: use_chirality = True
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def load_mol(molpath, explicit_H, use_chirality):
    if explicit_H is None: explicit_H = False
    if use_chirality is None: use_chirality = True
    # load mol
    if re.search(r'.pdb$', molpath):
        mol = Chem.MolFromPDBFile(molpath, removeHs=explicit_H, sanitize=False)
    elif re.search(r'.mol2$', molpath):
        mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H, sanitize=False)
    elif re.search(r'.sdf$', molpath):
        mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H, sanitize=False)
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    if mol is None:
        raise TypeError("Molecule is None in mol2graph_rdmda_res")
    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def mol_to_graph(mol, explicit_H, use_chirality):
    if explicit_H is None: explicit_H = False
    if use_chirality is None: use_chirality = True
    """
	mol: rdkit.Chem.rdchem.Mol
	explicit_H: whether to use explicit H
	use_chirality: whether to use chirality
	"""
    # Add nodes
    num_atoms = mol.GetNumAtoms()

    atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    # obtain the positions of the atoms
    atomCoords = mol.GetConformer().GetPositions()

    # Add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
        # bond_feats = np.concatenate([[1],bond_feats])
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g = Data(x=th.tensor(atom_feats, dtype=th.float),
             edge_index=th.tensor([src_list, dst_list]),
             pos=th.tensor(atomCoords, dtype=th.float),
             edge_attr=th.tensor(np.array(bond_feats_all), dtype=th.float))

    return g


def mol_to_graph2(prot_path, lig_path, cutoff=10.0, explicit_H=False, use_chirality=True):
    if explicit_H is None: explicit_H = False
    if use_chirality is None: use_chirality = True
    prot = load_mol(prot_path, explicit_H=explicit_H, use_chirality=use_chirality)
    lig = load_mol(lig_path, explicit_H=explicit_H, use_chirality=use_chirality)
    gp = prot_to_graph(prot, cutoff)
    gl = mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality)
    return gp, gl


def label_query(pdbid, args):
    if args.refined_csv:
        energy = args.ref.loc[pdbid, 'energy_mean']
        return energy
    else:
        logKd = args.ref.loc[pdbid, "-logKd/Ki"]
        return logKd

def element_in_file(file_path, element):
    with open(file_path, 'r') as file:
        for line in file:
            if element in line:
                return True
    return False


def pdbbind_handle(pdbid, args):
    if args.is_davide_set:
        prot_path = "%s/%s/%s_pocket.ligen.pdb" % (args.dir, pdbid, pdbid)
    else:
        prot_path = "%s/%s/%s_pocket.pdb" % (args.dir, pdbid, pdbid)
    lig_path = "%s/%s/%s_ligand.mol2" % (args.dir, pdbid, pdbid)
    # If we work on the palermo set we use both the general and refined set, so we have to set the path accordingly.
    if args.refined_csv:
        if not element_in_file(args.refined_csv, pdbid):
            prot_path = "%s/%s/%s_pocket.pdb" % (args.general_set_dir, pdbid, pdbid)
            lig_path = "%s/%s/%s_ligand.mol2" % (args.general_set_dir, pdbid, pdbid)

    try:
        gp, gl = mol_to_graph2(prot_path,
                               lig_path,
                               cutoff=args.cutoff,
                               explicit_H=args.useH,
                               use_chirality=args.use_chirality)
    except:
        print("%s failed to generate the graph"%pdbid)
        gp, gl = None, None
        # gm = None
    return pdbid, gp, gl, label_query(pdbid, args)


def UserInput():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', default=".",
                   help='The directory of the refined set')
    p.add_argument('-dg', '--general_set_dir', default=".",
                   help='The directory of the general set')
    p.add_argument('-c', '--cutoff', type=float,
                   help='the cutoff to determine the pocket')
    p.add_argument('-o', '--outprefix', default="out",
                   help='The output bin file.')
    p.add_argument('-r', '--ref', default="/home/shenchao/pdbbind/pdbbind_2020_general.csv",
                   help='The reference file to query the label of the complex.')
    p.add_argument('-usH', '--useH', default=False, action="store_true",
                   help='whether to use the explicit H atoms.')
    p.add_argument('-uschi', '--use_chirality', default=False, action="store_true",
                   help='whether to use chirality.')
    p.add_argument('-p', '--parallel', default=False, action="store_true",
                   help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')
    p.add_argument('-fs' '--filter_set', help="The csv file from wich you want to get the filtered pdbids")
    p.add_argument('-rc', '--refined_csv', help="The csv file containing the list of refined proteins")
    p.add_argument('-ds', '--is_davide_set', help="Whether we are using the Davide set or not.", action="store_true")

    args = p.parse_args()
    return args

def get_pdbids(args):
    palermoSet = pd.read_csv(args.fs__filter_set)
    pdbids = [x for x in os.listdir(args.dir) if os.path.isdir("%s/%s" % (args.dir, x))]
    res = palermoSet[palermoSet["energy_mean"]< -10]["pdb"].tolist()
    if (any(elem in pdbids for elem in res) == False):
        raise Exception("Some pdbids were not found")
    return res

def main():
    args = UserInput()
    if args.fs__filter_set:
        # If we are working on the palermo set we need to filter the proteins
        pdbids = get_pdbids(args)
        a = 0
        prots = []
        for pdbid in pdbids:
            if not element_in_file(args.refined_csv, pdbid):
                a = a+1
                prots.append(pdbid)
        print("Not in pdbind", a)
    else:
        # If we work on the standard refine set
        pdbids = [x for x in os.listdir(args.dir) if os.path.isdir("%s/%s" % (args.dir, x))]

    print("Found ", len(pdbids), "proteins")
    args.ref = pd.read_csv(args.ref, index_col=0, header=0)
    if args.parallel:
        print("Starting in multi process mode")
        results = Parallel(n_jobs=-1)(delayed(pdbbind_handle)(pdbid, args) for pdbid in pdbids)
        # Create a partial function with the constant parameter
        # partial_task = partial(pdbbind_handle, args=args)
        # with Pool(processes=1) as pool:
        #    pool.imap(partial_task, pdbids, chunksize=math.ceil(len(pdbids) / 2))
    else:
        print("Starting in single process mode")
        results = []
        with alive_bar(len(pdbids)) as bar:
            for pdbid in pdbids:
                results.append(pdbbind_handle(pdbid, args))
                bar()
    results = list(filter(lambda x: x[1] is not None, results))
    print("N of proteins:  ", len(pdbids))
    print("\033[31m", "N of failed prots: ", len(pdbids) - len(results), "\033[0m")
    print("\033[32m", "N of converted prots: ", len(results), "\033[0m")
    ids, graphs_p, graphs_l, labels = list(zip(*results))
    np.save("%s_ids" % args.outprefix, (ids, labels))
    th.save(graphs_p, "%s_prot.pt" % args.outprefix)
    th.save(graphs_l, "%s_lig.pt" % args.outprefix)


if __name__ == '__main__':
    main()
