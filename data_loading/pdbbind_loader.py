"""
PDBBind (v2020) data loader for ESA.

Constructs protein-ligand complex graphs from PDBBind refined set:
  - Nodes: atoms of the complex (protein + ligand)
  - Edges: inter-atomic contacts within a distance cutoff (default 6A)
  - Edge features: Gaussian-expanded inter-atomic distance + contact type
  - Target: binding affinity (pKd / pKi / pIC50), regression task
  - Caching: built graphs saved to <parent_dir>/pdbbind_cache.pt for fast reload

Data source: http://www.pdbbind.org.cn/ (v2020 refined set)

Directory structure expected:
  <root>/
    index/INDEX_refined_data.2020  (or INDEX_refined_data.2020 in root)
    <PDB_ID>/
      <PDB_ID>_protein.pdb
      <PDB_ID>_ligand.mol2
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count

from data_loading.transforms import (
    AddMaxEdge,
    AddMaxNode,
    AddMaxEdgeGlobal,
    AddMaxNodeGlobal,
    FormatSingleLabel,
)

# --- Config ---
MAX_ATOMS = 1500      # skip complexes larger than this
CUTOFF = 6.0           # edge distance cutoff in Angstrom
NUM_GAUSSIANS = 64     # number of Gaussian basis for edge features


# --- Biopython-free PDB coordinate reader ---

def parse_pdb_coords(pdb_path):
    """Parse atomic coordinates from a PDB file. Returns (coords, atom_types, residue_ids)."""
    coords = []
    atom_types = []
    residue_ids = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atom = line[76:78].strip() or line[12:16].strip()
                    resi = line[22:27].strip()
                    coords.append([x, y, z])
                    atom_types.append(atom)
                    residue_ids.append(resi)
                except (ValueError, IndexError):
                    continue
    return np.array(coords, dtype=np.float32), atom_types, residue_ids


def parse_mol2_coords(mol2_path):
    """Parse atomic coordinates from a Tripos MOL2 file."""
    coords = []
    atom_types = []
    with open(mol2_path, "r") as f:
        lines = f.readlines()
    in_atom = False
    for line in lines:
        if "@<TRIPOS>ATOM" in line:
            in_atom = True
            continue
        if "@<TRIPOS>BOND" in line:
            break
        if in_atom:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atom = parts[5].split(".")[0]
                    coords.append([x, y, z])
                    atom_types.append(atom)
                except (ValueError, IndexError):
                    continue
    return np.array(coords, dtype=np.float32), atom_types


# --- Element feature vocabulary ---

ELEMENT_VOCAB = {
    "H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "P": 5, "F": 6,
    "Cl": 7, "Br": 8, "I": 9, "Zn": 10, "Mg": 11, "Ca": 12,
    "Fe": 13, "Mn": 14, "Cu": 15, "Na": 16, "K": 17, "OTHER": 18,
}
NUM_ELEMENTS = len(ELEMENT_VOCAB)
ATOM_FEATURE_DIM = NUM_ELEMENTS + 1


def _atom_to_feature(atom_type, is_ligand):
    one_hot = [0.0] * NUM_ELEMENTS
    idx = ELEMENT_VOCAB.get(atom_type.upper(), ELEMENT_VOCAB["OTHER"])
    one_hot[idx] = 1.0
    return one_hot + [1.0 if is_ligand else 0.0]


# --- Gaussian basis for distance encoding ---

def gaussian_expansion(distances, min_dist=0.0, max_dist=6.0, num_gaussians=64):
    """Expand distances into Gaussian basis."""
    centers = np.linspace(min_dist, max_dist, num_gaussians)
    widths = (max_dist - min_dist) / (num_gaussians - 1)
    distances = np.expand_dims(distances, axis=-1)
    centers = np.expand_dims(centers, axis=0)
    return np.exp(-0.5 * ((distances - centers) / widths) ** 2)


# --- Single graph construction (standalone for multiprocessing) ---

def _build_single_graph(args):
    """
    Build one PDBBind graph. Standalone function for multiprocessing.
    Returns (Data or None, pdb_id).
    """
    complex_dir, pdb_id, affinity = args
    protein_pdb = os.path.join(complex_dir, f"{pdb_id}_protein.pdb")
    ligand_mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")

    if not os.path.exists(protein_pdb) or not os.path.exists(ligand_mol2):
        return None, pdb_id

    prot_coords, prot_atoms, _ = parse_pdb_coords(protein_pdb)
    lig_coords, lig_atoms = parse_mol2_coords(ligand_mol2)

    if len(prot_coords) == 0 or len(lig_coords) == 0:
        return None, pdb_id

    total_n = len(prot_coords) + len(lig_coords)
    if total_n > MAX_ATOMS:
        return None, pdb_id

    n_prot = len(prot_coords)
    all_coords = np.vstack([prot_coords, lig_coords])

    # Node features
    node_features = []
    for atom in prot_atoms:
        node_features.append(_atom_to_feature(atom, is_ligand=False))
    for atom in lig_atoms:
        node_features.append(_atom_to_feature(atom, is_ligand=True))
    x = torch.tensor(node_features, dtype=torch.float)

    # Build edges: cross protein-ligand + within-ligand, skip protein-protein
    edge_indices = []
    edge_dists = []

    for i in range(total_n):
        start_j = 0 if i >= n_prot else n_prot
        for j in range(start_j, total_n):
            if i >= j:
                continue
            d = np.linalg.norm(all_coords[i] - all_coords[j])
            if d < CUTOFF:
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                edge_dists.append(d)
                edge_dists.append(d)

    if len(edge_indices) == 0:
        # Fallback: nearest neighbor
        for i in range(n_prot, total_n):
            dists = np.linalg.norm(all_coords[:n_prot] - all_coords[i], axis=1)
            nearest = np.argmin(dists)
            edge_indices.append([i, nearest])
            edge_indices.append([nearest, i])
            edge_dists.append(dists[nearest])
            edge_dists.append(dists[nearest])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).T
    edge_dists = np.array(edge_dists, dtype=np.float32)
    edge_attr = torch.tensor(gaussian_expansion(edge_dists, 0.0, CUTOFF, NUM_GAUSSIANS), dtype=torch.float)

    y = torch.tensor([[affinity]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.max_node = total_n
    data.max_edge = edge_index.shape[1]

    return data, pdb_id


# --- Main loader with caching ---

def load_pdbbind(dataset_dir, one_hot=True, target_name=None, **kwargs):
    """
    Load PDBBind refined set with caching.

    Args:
        dataset_dir: Path to PDBBind v2020 refined set

    Returns:
        train, val, test datasets, num_classes, task_type, scaler
    """
    cache_path = os.path.join(os.path.dirname(os.path.abspath(dataset_dir)), "pdbbind_cache.pt")

    # Fast path: load from cache
    if os.path.exists(cache_path):
        print(f"Loading cached PDBBind graphs from {cache_path}...")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        max_nodes = cache["max_nodes"]
        max_edges = cache["max_edges"]
        scaler = cache["scaler"]

        def rebuild(samples, mx_n, mx_e):
            t = Compose([AddMaxEdgeGlobal(mx_e), AddMaxNodeGlobal(mx_n)])
            data_list = []
            for x, ei, ea, y in tqdm(samples, desc="Rebuilding from cache"):
                d = Data(x=x, edge_index=ei, edge_attr=ea, y=y)
                d.max_node = mx_n
                d.max_edge = mx_e
                d = t(d)
                data_list.append(d)
            return _wrap_dataset(data_list)

        train_ds = rebuild(cache["train"], max_nodes, max_edges)
        val_ds = rebuild(cache["val"], max_nodes, max_edges)
        test_ds = rebuild(cache["test"], max_nodes, max_edges)
        n_train = len(cache["train"])
        n_val = len(cache["val"])
        n_test = len(cache["test"])
        print(f"Loaded {n_train} train / {n_val} val / {n_test} test graphs (cached)")
        return train_ds, val_ds, test_ds, 1, "regression", scaler

    # Slow path: build from scratch
    # Find index file
    index_path = os.path.join(dataset_dir, "INDEX_refined_data.2020")
    alt_paths = [
        os.path.join(dataset_dir, "INDEX_general_PL_data.2020"),
        os.path.join(dataset_dir, "INDEX_refined_data.2016"),
        os.path.join(dataset_dir, "index", "INDEX_refined_data.2020"),
    ]
    if not os.path.exists(index_path):
        for ap in alt_paths:
            if os.path.exists(ap):
                index_path = ap
                break
        else:
            raise FileNotFoundError(
                f"No PDBBind index file found in {dataset_dir}."
            )

    # Parse index
    rows = []
    with open(index_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                pdb_id = parts[0]
                try:
                    affinity = float(parts[3])
                except ValueError:
                    continue
                rows.append({"pdb_id": pdb_id, "pKd": affinity})

    index_df = pd.DataFrame(rows)
    print(f"Found {len(index_df)} complexes in PDBBind index.")

    # Split
    index_df = index_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_total = len(index_df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_df = index_df.iloc[:n_train]
    val_df = index_df.iloc[n_train:n_train + n_val]
    test_df = index_df.iloc[n_train + n_val:]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Prepare args for multiprocessing
    def df_to_args(df):
        args_list = []
        for _, row in df.iterrows():
            pdb_id = row["pdb_id"]
            complex_dir = os.path.join(dataset_dir, pdb_id)
            if not os.path.isdir(complex_dir):
                for alt in [pdb_id.lower(), pdb_id.upper()]:
                    alt_dir = os.path.join(dataset_dir, alt)
                    if os.path.isdir(alt_dir):
                        complex_dir = alt_dir
                        break
                else:
                    continue
            args_list.append((complex_dir, pdb_id, row["pKd"]))
        return args_list

    train_args = df_to_args(train_df)
    val_args = df_to_args(val_df)
    test_args = df_to_args(test_df)

    # Build graphs with multiprocessing
    num_workers = min(8, cpu_count())
    print(f"\nBuilding graphs with {num_workers} workers (max {MAX_ATOMS} atoms per graph)...")

    def build_split(args_list, split_name):
        if len(args_list) == 0:
            return []
        data_list = []
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_build_single_graph, args_list),
                total=len(args_list),
                desc=f"Building {split_name}"
            ))
        for data, pdb_id in results:
            if data is not None:
                data_list.append(data)
        print(f"  {split_name}: {len(data_list)} / {len(args_list)} complexes built")
        return data_list

    train_list = build_split(train_args, "train")
    val_list = build_split(val_args, "val")
    test_list = build_split(test_args, "test")

    total_built = len(train_list) + len(val_list) + len(test_list)
    print(f"\nBuilt {len(train_list)} train / {len(val_list)} val / {len(test_list)} test graphs (total: {total_built})")

    if len(train_list) == 0:
        raise RuntimeError("No graphs could be built. Check PDBBind directory structure.")

    # Determine max nodes/edges
    max_nodes = max(d.max_node for d in train_list + val_list + test_list)
    max_edges = max(d.max_edge for d in train_list + val_list + test_list)
    print(f"Max nodes: {max_nodes}, Max edges: {max_edges}")

    # Apply padding transforms
    tf = Compose([AddMaxEdgeGlobal(max_edges), AddMaxNodeGlobal(max_nodes)])
    train_list = [tf(d) for d in tqdm(train_list, desc="Padding train")]
    val_list = [tf(d) for d in tqdm(val_list, desc="Padding val")]
    test_list = [tf(d) for d in tqdm(test_list, desc="Padding test")]

    # Scale targets
    scaler = StandardScaler()
    y_train = np.array([d.y.squeeze().item() for d in train_list], dtype=float).reshape(-1, 1)
    scaler.fit(y_train)

    def apply_scaler(data):
        data.y = torch.tensor(scaler.transform(data.y.reshape(1, -1).numpy()))
        return data

    train_list = [apply_scaler(d) for d in tqdm(train_list, desc="Scaling train")]
    val_list = [apply_scaler(d) for d in tqdm(val_list, desc="Scaling val")]
    test_list = [apply_scaler(d) for d in tqdm(test_list, desc="Scaling test")]

    # Wrap as datasets
    train_ds = _wrap_dataset(train_list)
    val_ds = _wrap_dataset(val_list)
    test_ds = _wrap_dataset(test_list)

    # Save cache
    torch.save({
        "train": [(d.x, d.edge_index, d.edge_attr, d.y) for d in train_list],
        "val": [(d.x, d.edge_index, d.edge_attr, d.y) for d in val_list],
        "test": [(d.x, d.edge_index, d.edge_attr, d.y) for d in test_list],
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "scaler": scaler,
    }, cache_path)
    print(f"Saved cache to {cache_path}")

    return train_ds, val_ds, test_ds, 1, "regression", scaler


class _PDBBindInMemory(InMemoryDataset):
    """Minimal InMemoryDataset wrapper for PDBBind data list."""
    def __init__(self, data_list):
        super().__init__(".")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def _wrap_dataset(data_list):
    return _PDBBindInMemory(data_list)


# --- Custom loader export ---

def load_pdbbind_function(dataset_dir, **kwargs):
    """Compatible interface for ESA's get_dataset_train_val_test."""
    return load_pdbbind(dataset_dir, **kwargs)
