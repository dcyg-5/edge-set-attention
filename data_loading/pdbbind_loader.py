"""
PDBBind (v2020) data loader for ESA.

Constructs protein-ligand complex graphs from PDBBind refined set:
  - Nodes: atoms of the complex (protein + ligand)
  - Edges: inter-atomic contacts within a distance cutoff (default 6A)
  - Edge features: Gaussian-expanded inter-atomic distance + contact type
  - Target: binding affinity (pKd / pKi / pIC50), regression task

Data source: http://www.pdbbind.org.cn/ (v2020 refined set)

Directory structure expected:
  <root>/
    INDEX_refined_data.2020  (or similar index)
    <PDB_ID>/
      <PDB_ID>_protein.pdb
      <PDB_ID>_ligand.mol2
      <PDB_ID>_pocket.pdb   (optional, residues within ~10A of ligand)
      <PDB_ID>_min.sdf       (optimised ligand geometry)
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose
from sklearn.preprocessing import StandardScaler

from data_loading.transforms import (
    AddMaxEdge,
    AddMaxNode,
    AddMaxEdgeGlobal,
    AddMaxNodeGlobal,
    FormatSingleLabel,
)


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
                    atom = parts[5].split(".")[0]  # e.g. C.3 -> C
                    coords.append([x, y, z])
                    atom_types.append(atom)
                except (ValueError, IndexError):
                    continue
    return np.array(coords, dtype=np.float32), atom_types


# --- Element feature vocabulary (for node features) ---

ELEMENT_VOCAB = {
    "H": 0, "C": 1, "N": 2, "O": 3, "S": 4, "P": 5, "F": 6,
    "Cl": 7, "Br": 8, "I": 9, "Zn": 10, "Mg": 11, "Ca": 12,
    "Fe": 13, "Mn": 14, "Cu": 15, "Na": 16, "K": 17, "OTHER": 18,
}
NUM_ELEMENTS = len(ELEMENT_VOCAB)
ATOM_FEATURE_DIM = NUM_ELEMENTS + 1  # one-hot + is_ligand


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
    distances = np.expand_dims(distances, axis=-1)  # N x 1
    centers = np.expand_dims(centers, axis=0)        # 1 x G
    return np.exp(-0.5 * ((distances - centers) / widths) ** 2)


# --- Main graph construction ---

def construct_pdbbind_graph(complex_dir, pdb_id, index_row, cutoff=6.0, num_gaussians=64):
    """
    Construct a PyG Data object for one PDBBind complex.

    Returns:
        Data with fields:
          - x: node features (atom type one-hot + ligand flag)
          - edge_index: [2, E] contact edges
          - edge_attr: [E, num_gaussians] distance features
          - y: binding affinity (pKd/pKi)
          - num_nodes, num_edges (for padding)
    """
    protein_pdb = os.path.join(complex_dir, f"{pdb_id}_protein.pdb")
    ligand_mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")

    if not os.path.exists(protein_pdb) or not os.path.exists(ligand_mol2):
        # Try alternative naming
        protein_pdb = os.path.join(complex_dir, f"{pdb_id}_protein.pdb")
        ligand_mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
        if not os.path.exists(protein_pdb) or not os.path.exists(ligand_mol2):
            return None

    # Parse coordinates
    prot_coords, prot_atoms, _ = parse_pdb_coords(protein_pdb)
    lig_coords, lig_atoms = parse_mol2_coords(ligand_mol2)

    if len(prot_coords) == 0 or len(lig_coords) == 0:
        return None

    # Combine protein + ligand
    all_coords = np.vstack([prot_coords, lig_coords])
    n_prot = len(prot_coords)
    n_lig = len(lig_coords)
    total_n = n_prot + n_lig

    # Node features: one-hot atom type + ligand flag
    node_features = []
    for atom in prot_atoms:
        node_features.append(_atom_to_feature(atom, is_ligand=False))
    for atom in lig_atoms:
        node_features.append(_atom_to_feature(atom, is_ligand=True))
    x = torch.tensor(node_features, dtype=torch.float)

    # Build edges: all atom pairs within cutoff distance
    # (only cross protein-ligand + within-ligand, skip protein-protein to keep graph manageable)
    edge_indices = []
    edge_dists = []

    for i in range(total_n):
        start_j = 0 if i >= n_prot else n_prot  # protein only connects to ligand, ligand connects to all
        for j in range(start_j, total_n):
            if i >= j:
                continue
            d = np.linalg.norm(all_coords[i] - all_coords[j])
            if d < cutoff:
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                edge_dists.append(d)
                edge_dists.append(d)

    if len(edge_indices) == 0:
        # Fallback: nearest neighbor connection for each ligand atom
        for i in range(n_prot, total_n):
            dists = np.linalg.norm(all_coords[:n_prot] - all_coords[i], axis=1)
            nearest = np.argmin(dists)
            edge_indices.append([i, nearest])
            edge_indices.append([nearest, i])
            edge_dists.append(dists[nearest])
            edge_dists.append(dists[nearest])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).T  # [2, E]

    # Edge features: Gaussian expansion of distances
    edge_dists = np.array(edge_dists, dtype=np.float32)
    edge_attr = torch.tensor(gaussian_expansion(edge_dists, 0.0, cutoff, num_gaussians), dtype=torch.float)

    # Target: binding affinity
    affinity_key = index_row.get("affinity_key", "pKd")
    if affinity_key in index_row and pd.notna(index_row[affinity_key]):
        y = torch.tensor([[float(index_row[affinity_key])]], dtype=torch.float)
    else:
        return None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Add metadata for padding (needed by ESA transforms)
    data.max_node = total_n
    data.max_edge = edge_index.shape[1]

    return data


# --- Main loader function ---

def load_pdbbind(dataset_dir, one_hot=True, target_name=None, **kwargs):
    """
    Load PDBBind refined set.

    Args:
        dataset_dir: Path to PDBBind v2020 refined set (contains 'INDEX_refined_data.2020')
        one_hot: Already true by default for PDBBind
        target_name: Ignored (PDBBind has only one target per entry)
    
    Returns:
        train, val, test datasets, num_classes, task_type, scaler
    """
    # Parse index file
    index_path = os.path.join(dataset_dir, "INDEX_refined_data.2020")
    
    if os.path.exists(index_path):
        # Standard PDBBind index format: column-based
        rows = []
        with open(index_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    pdb_id = parts[0]
                    resolution = parts[1]
                    release_year = parts[2]
                    # pKd/pKi/pIC50 is usually the last column
                    affinity_str = parts[-1]
                    try:
                        affinity = float(affinity_str)
                    except ValueError:
                        continue
                    rows.append({
                        "pdb_id": pdb_id,
                        "resolution": resolution,
                        "year": release_year,
                        "pKd": affinity,
                    })
        index_df = pd.DataFrame(rows)
        affinity_key = "pKd"
    else:
        # Try alternative: General PDBBind v2020 index files
        alt_paths = [
            os.path.join(dataset_dir, "INDEX_general_PL_data.2020"),
            os.path.join(dataset_dir, "INDEX_refined_data.2016"),
            os.path.join(dataset_dir, "index", "INDEX_refined_data.2020"),
        ]
        for ap in alt_paths:
            if os.path.exists(ap):
                index_path = ap
                break
        else:
            raise FileNotFoundError(
                f"No PDBBind index file found in {dataset_dir}. "
                f"Expected INDEX_refined_data.2020 or similar."
            )
        rows = []
        with open(index_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    pdb_id = parts[0]
                    try:
                        affinity = float(parts[-1])
                    except ValueError:
                        continue
                    rows.append({
                        "pdb_id": pdb_id,
                        "pKd": affinity,
                    })
        index_df = pd.DataFrame(rows)
        affinity_key = "pKd"

    print(f"Found {len(index_df)} complexes in PDBBind index.")

    # Shuffle and split
    index_df = index_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_total = len(index_df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_df = index_df.iloc[:n_train]
    val_df = index_df.iloc[n_train:n_train + n_val]
    test_df = index_df.iloc[n_train + n_val:]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Construct graphs
    def process_split(df, split_name):
        data_list = []
        for _, row in tqdm(df.iterrows(), desc=f"Processing {split_name}", total=len(df)):
            pdb_id = row["pdb_id"]
            complex_dir = os.path.join(dataset_dir, pdb_id)
            if not os.path.isdir(complex_dir):
                # Try uppercase/lowercase variants
                for alt in [pdb_id.lower(), pdb_id.upper()]:
                    alt_dir = os.path.join(dataset_dir, alt)
                    if os.path.isdir(alt_dir):
                        complex_dir = alt_dir
                        break
                else:
                    continue
            data = construct_pdbbind_graph(
                complex_dir, pdb_id, row, cutoff=6.0, num_gaussians=64
            )
            if data is not None:
                data_list.append(data)
        return data_list

    print("\nBuilding graphs...")
    train_list = process_split(train_df, "train")
    val_list = process_split(val_df, "val")
    test_list = process_split(test_df, "test")

    print(f"\nBuilt {len(train_list)} train / {len(val_list)} val / {len(test_list)} test graphs")

    if len(train_list) == 0:
        raise RuntimeError(
            "No graphs could be built. Check PDBBind directory structure:\n"
            f"  {dataset_dir}\n"
            "Expected subdirectories named by PDB ID, each containing\n"
            "  *_protein.pdb and *_ligand.mol2 files."
        )

    # Determine max nodes / max edges globally
    max_nodes = max(d.max_node for d in train_list + val_list + test_list)
    max_edges = max(d.max_edge for d in train_list + val_list + test_list)
    print(f"Max nodes per graph: {max_nodes}, Max edges per graph: {max_edges}")

    # Apply global transforms
    transforms = Compose([
        AddMaxEdgeGlobal(max_edges),
        AddMaxNodeGlobal(max_nodes),
    ])

    train_list = [transforms(d) for d in tqdm(train_list, desc="Transforming train")]
    val_list = [transforms(d) for d in tqdm(val_list, desc="Transforming val")]
    test_list = [transforms(d) for d in tqdm(test_list, desc="Transforming test")]

    # Scale regression target
    scaler = StandardScaler()
    y_train = np.array([d.y.squeeze().item() for d in train_list], dtype=float).reshape(-1, 1)
    scaler = scaler.fit(y_train)

    def apply_scaler(data):
        data.y = torch.tensor(scaler.transform(data.y.reshape(1, -1).numpy()))
        return data

    train_list = [apply_scaler(d) for d in tqdm(train_list, desc="Scaling train")]
    val_list = [apply_scaler(d) for d in tqdm(val_list, desc="Scaling val")]
    test_list = [apply_scaler(d) for d in tqdm(test_list, desc="Scaling test")]

    # Wrap as InMemoryDataset
    train_ds = _wrap_dataset(train_list)
    val_ds = _wrap_dataset(val_list)
    test_ds = _wrap_dataset(test_list)

    num_classes = 1
    task_type = "regression"

    print("PDBBind loading complete!")
    return train_ds, val_ds, test_ds, num_classes, task_type, scaler


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


# --- Custom loader export (for use in data_loading.py) ---

def load_pdbbind_function(dataset_dir, **kwargs):
    """Compatible interface for ESA's get_dataset_train_val_test."""
    return load_pdbbind(dataset_dir, **kwargs)
