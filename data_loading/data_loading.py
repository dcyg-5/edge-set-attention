import numpy as np
import pandas as pd
import os
import torch
import torch_geometric
from torch_geometric.utils import degree
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from ogb.utils import smiles2graph
from data_loading.pyg_molecular_datasets.qm9 import QM9 as CustomQM9
from data_loading.pyg_molecular_datasets.molecule_net import MoleculeNet as CustomMoleculeNet
from data_loading.lrgb import PeptidesFunctionalDataset, PeptidesStructuralDataset
from data_loading.graphgps_utils import join_dataset_splits
from data_loading.transforms import *

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

QM9_TARGETS = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
    "atom_u0",
    "atom_u298",
    "atom_h298",
    "atom_g298",
    "rot_constant_a",
    "rot_constant_b",
    "rot_constant_c",
]

DOCKSTRING_TARGETS = [
    "ESR2",
    "KIT",
    "PARP1",
    "PGR",
    "PARP1"
]

# 1 means binary classification
TUDATASET_NUM_CLASSES_DICT = {
    "ENZYMES": 6,
    "PROTEINS_full": 1,
    "DD": 1,
    "IMDB-BINARY": 1,
    "IMDB-MULTI": 3,
    "REDDIT-BINARY": 1,
    "REDDIT-MULTI-5K": 5,
    "REDDIT-MULTI-12K": 11,
    "reddit_threads": 1,
    "twitch_egos": 1,
    "github_stargazers": 1,
    "SYNTHETIC": 1,
    "SYNTHETICnew": 1,
    "Synthie": 4,
    "COLORS-3": 11,
    "TRIANGLES": 10,
    "NCI1": 1,
    "NCI109": 1,
}

MOLECULENET_NUM_CLASSES_DICT = {
    "ESOL": 1,
    "FreeSolv": 1,
    "Lipo": 1,
    "HIV": 1,
    "BACE": 1,
    "BBBP": 1,
}

DATASETS_WITH_INTEGER_FEATURES = [
    "QM9",
    "DOCKSTRING",
    "MalNetTiny",
    "DD",
    "github_stargazers",
    "IMDB-BINARY",
    "reddit_threads",
    "twitch_egos",
    "ESOL",
    "FreeSolv",
    "Lipo",
    "HIV",
    "BACE",
    "BBBP",
    "NCI1",
    "NCI109"
]

MOLECULENET_MAX_ATOMIC_NUMBERS = {
    "ESOL": 53,
    "FreeSolv": 53,
    "Lipo": 53,
    "HIV": 92,
    "BACE": 53,
    "BBBP": 53,
}


class CustomPyGDataset(InMemoryDataset):
    def __init__(self, data_list=None):
        super(CustomPyGDataset, self).__init__(".")
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class CustomPyGDatasetNodeMasks(InMemoryDataset):
    def __init__(self, data_list, train_mask, val_mask, test_mask):
        super(CustomPyGDatasetNodeMasks, self).__init__(".")
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)
            self.train_mask = train_mask
            self.val_mask = val_mask
            self.test_mask = test_mask

    def _download(self):
        pass

    def _process(self):
        pass


def apply_scaler(data, scaler, convert_to_numpy=True, num_tasks=1):
    if convert_to_numpy:
        data.y = torch.tensor(scaler.transform(data.y.reshape(1, num_tasks).numpy()))
    else:
        data.y = torch.tensor(scaler.transform(data.y.reshape(1, num_tasks)))
    return data


# Function to convert Subset to PyG Dataset
def subset_to_pyg_dataset(subset, scaler=None, num_tasks=1):
    if not scaler:
        data_list = [subset.dataset[i] for i in subset.indices]
    else:
        data_list = [apply_scaler(subset.dataset[i], scaler, num_tasks=num_tasks) for i in subset.indices]

    return CustomPyGDataset(data_list=data_list)


def filter_none(data):
    return data is not None


def scale_y_for_regression_task(train_data_list, num_tasks=1):
    y_train = np.array([data.y.squeeze() for data in train_data_list], dtype=float).reshape(-1, num_tasks)

    scaler = StandardScaler()
    scaler = scaler.fit(y_train)

    return scaler


def get_max_node_edge_global(dataset):
    max_node_global = 0
    max_edge_global = 0

    for data in tqdm(dataset):
        if data.max_edge > max_edge_global:
            max_edge_global = data.max_edge
        if data.max_node > max_node_global:
            max_node_global = data.max_node

    return max_edge_global, max_node_global


def load_malnettiny(download_dir, one_hot, **kwargs):
    if one_hot:
        degree_t = T.OneHotDegree(max_degree=1556)
    else:
        degree_t = OneHotInt(max_degree=1556)

    transforms = [T.RemoveIsolatedNodes(), degree_t, AddMaxEdge(), AddMaxNode(), AddMaxDegree(1556)]
    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    # Cannot use pre-transforms due to very high memory usage
    train = torch_geometric.datasets.MalNetTiny(
        root=download_dir,
        transform=T.Compose(transforms),
        split="train",
    )

    val = torch_geometric.datasets.MalNetTiny(
        root=download_dir,
        transform=T.Compose(transforms),
        split="val",
    )

    test = torch_geometric.datasets.MalNetTiny(
        root=download_dir,
        transform=T.Compose(transforms),
        split="test",
    )

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    print("Finished loading data!")

    num_classes = 5
    task_type = "multi_classification" if num_classes > 2 else "binary_classification"

    return train, val, test, num_classes, task_type, None


def load_gnn_benchmark(dataset, download_dir, **kwargs):
    transforms = [Add3DOrPosAsNodeFeatures(), AddMaxEdge(), AddMaxNode(), EdgeFeaturesUnsqueeze()]
    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    train = torch_geometric.datasets.GNNBenchmarkDataset(
        root=download_dir,
        name=dataset,
        split="train",
        pre_transform=T.Compose(transforms),
    )

    val = torch_geometric.datasets.GNNBenchmarkDataset(
        root=download_dir,
        name=dataset,
        split="val",
        pre_transform=T.Compose(transforms),
    )

    test = torch_geometric.datasets.GNNBenchmarkDataset(
        root=download_dir,
        name=dataset,
        split="test",
        pre_transform=T.Compose(transforms),
    )

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    print("Finished loading data!")

    num_classes = 10
    task_type = "multi_classification" if num_classes > 2 else "binary_classification"

    return train, val, test, num_classes, task_type, None


def load_tudataset(dataset_name, download_dir, one_hot=True, **kwargs):
    transforms = [AddMaxEdge(), AddMaxNode()]
    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    dataset = torch_geometric.datasets.TUDataset(
        root=download_dir,
        name=dataset_name,
        use_node_attr=True,
        use_edge_attr=True,
        pre_filter=filter_none,
        pre_transform=T.Compose(transforms),
    )

    if dataset_name in [
        "github_stargazers",
        "IMDB-BINARY",
        "IMDB-MULTI",
        "reddit_threads",
        "twitch_egos",
        "TRIANGLES",
        "NCI1",
        "NCI109",
    ]:
        degree_0 = np.max([np.max(degree(d.edge_index[0]).detach().cpu().numpy()) for d in dataset])
        degree_1 = np.max([np.max(degree(d.edge_index[1]).detach().cpu().numpy()) for d in dataset])

        max_degree = int(max(degree_0, degree_1))

        if one_hot:
            degree_t = T.OneHotDegree(max_degree=max_degree)
        else:
            degree_t = OneHotInt(max_degree=max_degree)

        PRE_TRANSFORMS = [degree_t, AddMaxEdge(), AddMaxNode(), AddMaxDegree(max_degree)]

        if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
            PRE_TRANSFORMS.append(t_posenc)

        dataset = torch_geometric.datasets.TUDataset(
            root=os.path.join(download_dir, "with_degrees"),
            name=dataset_name,
            use_node_attr=True,
            use_edge_attr=True,
            pre_transform=T.Compose(PRE_TRANSFORMS),
        )

    print("\nDataset items look like: ", dataset[0])

    max_edge_global, max_node_global = get_max_node_edge_global(dataset)

    print(f"Datasets has {len(dataset)} elements")
    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    dataset = [global_transforms(data) for data in tqdm(dataset)]

    train_len = int(len(dataset) * 0.8)
    val_len = int(len(dataset) * 0.1)
    test_len = len(dataset) - train_len - val_len

    train, val, test = random_split(
        dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    train = subset_to_pyg_dataset(train)
    val = subset_to_pyg_dataset(val)
    test = subset_to_pyg_dataset(test)

    print("Finished loading data!")

    if dataset_name in ["ENZYMES", "COLORS-3", "Synthie", "TRIANGLES", "IMDB-MULTI", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K"]:
        task_type = "multi_classification"
    else:
        task_type = "binary_classification"

    num_classes = TUDATASET_NUM_CLASSES_DICT[dataset_name]

    return train, val, test, num_classes, task_type, None


def load_qm9_chemprop(download_dir, one_hot, target_name, **kwargs):
    transforms = [
        SelectTarget(QM9_TARGETS.index(target_name)),
        ChempropFeatures(one_hot=one_hot, max_atomic_number=9),
        AddMaxEdge(),
        AddMaxNode(),
    ]

    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    transforms.append(FormatSingleLabel())

    dataset = CustomQM9(root=download_dir, pre_transform=T.Compose(transforms), pre_filter=filter_none)

    print("\nDataset items look like: ", dataset[0])

    print("Determining global node/edge counts...")

    max_edge_global, max_node_global = get_max_node_edge_global(dataset)

    print(f"Datasets has {len(dataset)} elements")
    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    dataset = [global_transforms(data) for data in tqdm(dataset)]

    train_len = int(len(dataset) * 0.8)
    val_len = int(len(dataset) * 0.1)
    test_len = len(dataset) - train_len - val_len

    train, val, test = random_split(
        dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    print("Scaling dataset y values...")

    y_scaler = scale_y_for_regression_task(train)

    train = subset_to_pyg_dataset(train, scaler=y_scaler)
    val = subset_to_pyg_dataset(val, scaler=y_scaler)
    test = subset_to_pyg_dataset(test, scaler=y_scaler)

    print("Finished loading data!")

    num_classes = 1
    task_type = "regression"

    return train, val, test, num_classes, task_type, y_scaler


def load_moleculenet_chemprop(dataset_name, download_dir, one_hot, **kwargs):
    transforms = [
        ChempropFeatures(one_hot=one_hot, max_atomic_number=MOLECULENET_MAX_ATOMIC_NUMBERS[dataset_name]),
        AddMaxEdge(),
        AddMaxNode(),
        FormatSingleLabel(),
        LabelNanToZero(),
    ]

    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    dataset = CustomMoleculeNet(
        root=download_dir, name=dataset_name, pre_transform=T.Compose(transforms), pre_filter=filter_none
    )

    print("\nDataset items look like: ", dataset[0])

    max_edge_global, max_node_global = get_max_node_edge_global(dataset)

    print(f"Datasets has {len(dataset)} elements")
    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    dataset = [global_transforms(data) for data in tqdm(dataset)]

    train_len = int(len(dataset) * 0.8)
    val_len = int(len(dataset) * 0.1)
    test_len = len(dataset) - train_len - val_len

    train, val, test = random_split(
        dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    if dataset_name in ["ESOL", "FreeSolv", "Lipo"]:
        print("Scaling dataset y values...")
        y_scaler = scale_y_for_regression_task(train)
    else:
        y_scaler = None

    train = subset_to_pyg_dataset(train, scaler=y_scaler)
    val = subset_to_pyg_dataset(val, scaler=y_scaler)
    test = subset_to_pyg_dataset(test, scaler=y_scaler)

    num_classes = MOLECULENET_NUM_CLASSES_DICT[dataset_name]
    if dataset_name in ["ESOL", "FreeSolv", "Lipo"]:
        task_type = "regression"
    else:
        task_type = "binary_classification"

    print("Finished loading data!")

    return train, val, test, num_classes, task_type, y_scaler


def load_dockstring_chemprop(dataset_dir, one_hot, target_name, **kwargs):
    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        posenc_name = "+".join(kwargs["pe_types"])
    else:
        posenc_name = "None"

    preprocessed_path_train = os.path.join(dataset_dir, f"DOCKSTRING_train_{target_name}_{posenc_name}.pt")
    preprocessed_path_val = os.path.join(dataset_dir, f"DOCKSTRING_val_{target_name}_{posenc_name}.pt")
    preprocessed_path_test = os.path.join(dataset_dir, f"DOCKSTRING_test_{target_name}_{posenc_name}.pt")

    if os.path.isfile(preprocessed_path_train):
        print("Loading pre-processed train, val, test...")
        train = torch.load(preprocessed_path_train)
        val = torch.load(preprocessed_path_val)
        test = torch.load(preprocessed_path_test)
        print("Loaded pre-processed splits!")
    else:
        def csv_to_pyg_data(df):
            dataset_as_data_list = []

            smiles = df["SMILES"].values
            y = df[target_name].values

            for i in tqdm(range(len(smiles))):
                dataset_as_data_list.append(Data(smiles=smiles[i], y=y[i]))

            return dataset_as_data_list

        print("Pre-processed data unavailable, computing...")
        train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
        val = pd.read_csv(os.path.join(dataset_dir, "val.csv"))
        test = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

        print("Loading data splits...")
        train = csv_to_pyg_data(train)
        val = csv_to_pyg_data(val)
        test = csv_to_pyg_data(test)

        print("\nDataset items look like: ", train[0])

        transforms = [ChempropFeatures(one_hot=one_hot, max_atomic_number=53), AddNumNodes(), AddMaxEdge(), AddMaxNode()]

        if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
            t_posenc = AddPosEnc(kwargs["pe_types"])
            transforms.append(t_posenc)

        transforms = T.Compose(transforms)

        print("Computing ChemProp features for data splits...")
        train = [transforms(data) for data in tqdm(train) if data is not None]
        val = [transforms(data) for data in tqdm(val) if data is not None]
        test = [transforms(data) for data in tqdm(test) if data is not None]

        print("Caching pre-processed files...")
        torch.save(train, preprocessed_path_train)
        torch.save(val, preprocessed_path_val)
        torch.save(test, preprocessed_path_test)
        print("Caching done")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"max_node_global = {max_node_global}")
    print(f"max_edge_global = {max_edge_global}")
    print(f"dataset size = {len(train) + len(val) + len(test)}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    y_scaler = scale_y_for_regression_task(train)

    print("Applying label scaler for data splits...")
    train = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(train)]
    val = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(val)]
    test = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(test)]

    train = CustomPyGDataset(train)
    val = CustomPyGDataset(val)
    test = CustomPyGDataset(test)

    print("Finished loading data!")

    num_classes = 1
    task_type = "regression"

    return train, val, test, num_classes, task_type, y_scaler


def load_lrgb_pept_fn(download_dir, **kwargs):
    transforms = [AddMaxEdge(), AddMaxNode(), OneHotYToSingle()]

    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    dataset = PeptidesFunctionalDataset(
        root=download_dir,
        smiles2graph=smiles2graph,
        pre_transform=T.Compose(transforms)
    )

    split_dict = dataset.get_idx_split()

    train = dataset[split_dict["train"]]
    val = dataset[split_dict["val"]]
    test = dataset[split_dict["test"]]

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]


    train = CustomPyGDataset(train)
    val = CustomPyGDataset(val)
    test = CustomPyGDataset(test)

    print("Finished loading data!")

    num_classes = 10
    task_type = "multi_classification"

    return train, val, test, num_classes, task_type, None


def load_lrgb_pept_struct(download_dir, **kwargs):
    transforms = [AddMaxEdge(), AddMaxNode()]

    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    dataset = PeptidesStructuralDataset(
        root=download_dir,
        smiles2graph=smiles2graph,
        pre_transform=T.Compose(transforms)
    )

    split_dict = dataset.get_idx_split()

    train = dataset[split_dict["train"]]
    val = dataset[split_dict["val"]]
    test = dataset[split_dict["test"]]

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    train = CustomPyGDataset(train)
    val = CustomPyGDataset(val)
    test = CustomPyGDataset(test)

    print("Finished loading data!")

    num_classes = 11
    task_type = "regression"

    return train, val, test, num_classes, task_type, None


def load_zinc_benchmark(dataset, download_dir, **kwargs):
    transforms = [AddMaxEdge(), AddMaxNode(), EdgeFeaturesUnsqueeze()]
    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    train = torch_geometric.datasets.ZINC(
        root=download_dir,
        subset=False,
        split="train",
        pre_transform=T.Compose(transforms),
    )

    val = torch_geometric.datasets.ZINC(
        root=download_dir,
        subset=False,
        split="val",
        pre_transform=T.Compose(transforms),
    )

    test = torch_geometric.datasets.ZINC(
        root=download_dir,
        subset=False,
        split="test",
        pre_transform=T.Compose(transforms),
    )

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    y_scaler = scale_y_for_regression_task(train)

    print("Applying label scaler for data splits...")
    train = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(train)]
    val = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(val)]
    test = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(test)]

    train = CustomPyGDataset(train)
    val = CustomPyGDataset(val)
    test = CustomPyGDataset(test)

    print("Finished loading data!")

    num_classes = 1
    task_type = "regression"

    return train, val, test, num_classes, task_type, y_scaler


def load_pcqm(download_dir, **kwargs):
    # NOTE: We disabled the code for the maximum number of nodes/edges for efficiency reasons (takes 30+ min)
    # NOTE: Instead, we hard-code this value in the ESA code to 118 (maximum number of edges)
    transforms = []

    if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
        t_posenc = AddPosEnc(kwargs["pe_types"])
        transforms.append(t_posenc)

    train = torch_geometric.datasets.PCQM4Mv2(
        root=download_dir,
        split="train",
        transform=T.Compose(transforms) if len(transforms) > 0 else None,
    )

    val = torch_geometric.datasets.PCQM4Mv2(
        root=download_dir,
        split="val",
        transform=T.Compose(transforms) if len(transforms) > 0 else None,
    )

    # test = torch_geometric.datasets.PCQM4Mv2(
    #     root=download_dir,
    #     split="test",
    #     transform=T.Compose(transforms) if len(transforms) > 0 else None,
    # )

    # test_holdout = torch_geometric.datasets.PCQM4Mv2(
    #     root=download_dir,
    #     split="holdout",
    #     transform=T.Compose(transforms) if len(transforms) > 0 else None,
    # )

    print("\nDataset items look like: ", train[0])

    print(f"Dataset has {len(train)} train elements")
    print(f"Dataset has {len(val)} validation elements")
    # print(f"Dataset has {len(test)} test elements")
    # print(f"Dataset has {len(test_holdout)} test_holdout elements")

    print("Finished loading data!")

    num_classes = 1
    task_type = "regression"

    return train, val, val, val, num_classes, task_type, None


def load_ppi(download_dir, **kwards):
    train = torch_geometric.datasets.PPI(root=download_dir, split="train", pre_transform=T.Compose([AddMaxEdge(), AddMaxNode()]))
    val = torch_geometric.datasets.PPI(root=download_dir, split="val", pre_transform=T.Compose([AddMaxEdge(), AddMaxNode()]))
    test = torch_geometric.datasets.PPI(root=download_dir, split="test", pre_transform=T.Compose([AddMaxEdge(), AddMaxNode()]))

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    print("Finished loading data!")
    
    num_classes = 2
    task_type = "binary_classification"

    return train, val, test, num_classes, task_type, None, None, None, None


def load_planetoid(dataset, download_dir, **kwargs):
    dataset = torch_geometric.datasets.Planetoid(
        root=download_dir,
        name=dataset,
        split="public",
        pre_transform=T.Compose([AddMaxEdge(), AddMaxNode()])
    )

    num_classes = dataset.num_classes
    task_type = "multi_classification"

    data = dataset[0]

    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    train = [data]
    val = [data]
    test = [data]

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    train = CustomPyGDatasetNodeMasks(train, train_mask, val_mask, test_mask)
    val = CustomPyGDatasetNodeMasks(val, train_mask, val_mask, test_mask)
    test = CustomPyGDatasetNodeMasks(test, train_mask, val_mask, test_mask)

    print("Finished loading data!")

    return train, val, test, num_classes, task_type, None, train_mask, val_mask, test_mask


def load_infected(dataset, dataset_dir, **kwargs):
    if dataset == "infected+15000":
        dataset = torch.load(os.path.join(dataset_dir, "ER+numn=15000+edge_p=0.00009+num_inf=40+max_p=20.pt"))
    elif dataset == "infected+30000":
        dataset = torch.load(os.path.join(dataset_dir,"ER+numn=30000+edge_p=0.00005+num_inf=20+max_p=20.pt"))

    pre_transform = T.Compose([AddMaxEdge(), AddMaxNode()])
    data = pre_transform(dataset[0])

    num_classes = dataset.num_classes
    task_type = "multi_classification"

    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    train = [data]
    val = [data]
    test = [data]

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global), AddMasks(train_mask, val_mask, test_mask)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    train = CustomPyGDatasetNodeMasks(train, train_mask, val_mask, test_mask)
    val = CustomPyGDatasetNodeMasks(val, train_mask, val_mask, test_mask)
    test = CustomPyGDatasetNodeMasks(test, train_mask, val_mask, test_mask)

    print("Finished loading data!")

    return train, val, test, num_classes, task_type, None, train_mask, val_mask, test_mask


def load_heterophilic(dataset, dataset_dir, **kwargs):
    hetero_name = dataset.split("+")[-1]
    # Originally from https://github.com/yandex-research/heterophilous-graphs
    dataset = torch.load(os.path.join(dataset_dir, f"{hetero_name}.pt"))

    data = dataset["data"]
    pre_transform = T.Compose([AddMaxEdge(), AddMaxNode()])
    data = pre_transform(data)

    num_classes = dataset["num_classes"]
    num_tasks = dataset["num_classes"]
    task_type = "multi_classification"

    train_mask = dataset["train_masks"][0]
    val_mask = dataset["val_masks"][0]
    test_mask = dataset["test_masks"][0]

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    train = [data]
    val = [data]
    test = [data]

    print("\nDataset items look like: ", train[0])

    print(f"Datasets has {len(train)} train elements")
    print(f"Datasets has {len(val)} validation elements")
    print(f"Datasets has {len(test)} test elements")

    print("Determining global node/edge counts...")
    max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
    max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
    max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)

    max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
    max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global), AddMasks(train_mask, val_mask, test_mask)])

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    train = CustomPyGDatasetNodeMasks(train, train_mask, val_mask, test_mask)
    val = CustomPyGDatasetNodeMasks(val, train_mask, val_mask, test_mask)
    test = CustomPyGDatasetNodeMasks(test, train_mask, val_mask, test_mask)

    print("Finished loading data!")

    return train, val, test, num_classes, task_type, None, train_mask, val_mask, test_mask


def get_dataset_train_val_test(dataset, dataset_dir, **kwargs):
    if dataset == "MalNetTiny":
        return load_malnettiny(dataset_dir, **kwargs)
    elif dataset in ["MNIST", "CIFAR10"]:
        return load_gnn_benchmark(dataset, dataset_dir, **kwargs)
    elif dataset == "QM9":
        return load_qm9_chemprop(dataset_dir, **kwargs)
    elif dataset == "DOCKSTRING":
        return load_dockstring_chemprop(dataset_dir, **kwargs)
    elif dataset in TUDATASET_NUM_CLASSES_DICT.keys():
        return load_tudataset(dataset, dataset_dir, **kwargs)
    elif dataset in MOLECULENET_NUM_CLASSES_DICT.keys():
        return load_moleculenet_chemprop(dataset, dataset_dir, **kwargs)
    elif dataset == "lrgb-pept-fn":
        return load_lrgb_pept_fn(dataset_dir, **kwargs)
    elif dataset == "lrgb-pept-struct":
        return load_lrgb_pept_struct(dataset_dir, **kwargs)
    elif dataset == "ZINC":
        return load_zinc_benchmark(dataset, dataset_dir, **kwargs)
    elif dataset == "PCQM4Mv2":
        return load_pcqm(dataset_dir, **kwargs)
    elif dataset == "PPI":
        return load_ppi(dataset_dir, **kwargs)
    elif dataset in ["Cora", "CiteSeer"]:
        return load_planetoid(dataset, dataset_dir, **kwargs)
    elif "infected" in dataset:
        return load_infected(dataset, dataset_dir, **kwargs)
    elif "hetero" in dataset:
        return load_heterophilic(dataset, dataset_dir, **kwargs)
    

def get_dataset_train_val_test_with_indices_for_graphgps(dataset, dataset_dir, **kwargs):
    train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(dataset, dataset_dir, **kwargs)
    joined = join_dataset_splits((train, val, test))
    return joined, scaler

def get_dataset_train_val_test_with_indices_for_graphgps_node(dataset, dataset_dir, **kwargs):
    # Train, val, test masks are attributes of the train object
    train, val, test, num_classes, task_type, scaler, train_mask, val_mask, test_mask = get_dataset_train_val_test(dataset, dataset_dir, **kwargs)
    return train, scaler
