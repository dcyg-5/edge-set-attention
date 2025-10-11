
  

# An end-to-end attention-based approach for learning on graphs

This source code archive accompanies the [An end-to-end attention-based approach for learning on graphs](https://www.nature.com/articles/s41467-025-60252-z) paper, out now in Nature Communications. It includes the code required to run experiments involving GNNs, ESA, Graphormer/TokenGT, GraphGPS, as well as instructions and all the used datasets.

The code was refactored several times during development and the results might slightly differ from the paper. This code is meant to act as a demo for the main experiments in the paper (benchmarks) and document the implementation.

Please read the detailed description for each argument to make sure that the code is setup correctly.

Currently, we do not explicitly include the time/memory tracking functionality. This can be implemented in a few lines of code but would needlessly complicate the provided code. However, we provide instructions at the end of this README on how to easily implement it.

We provide code for three main types of tasks:

1. Graph-level
2. Node-level
3. Transfer learning (including support for 3D atomic coordinates)
4. These are separated since the algorithms require different implementation strategies.

# Data

Most datasets used in this work can be downloaded through PyTorch Geometric (PyG). For those that are not available in PyG, we include links to access them, namely:

- DOCKSTRING
- QM9 with accurate GW/DFT frontier orbital energies and with 3D coordinates (for transfer learning)
- Shortest path datasets (infected) - these can be generated through PyG using the commands given in the SI of the paper. However, we provide them for convenience.
- Heterophily datasets (`roman empire`, `amazon ratings`, `minesweeper`, `tolokers`, `squirrel_filtered`, `chameleon_filtered`)

These can be downloaded from [this auxiliary repository](https://github.com/davidbuterez/esa-paper-extra-datasets) (remember to use `git lfs pull` after `git clone`). The datasets are already in the required format and are ready to be loaded using the data loading pipeline.

The only exception is for the Open Catalyst Project (OCP) dataset, as the files would total around 4GB. For OCP, the following files are required from https://fair-chem.github.io/core/datasets/oc20.html:

- Train: `.../is2re/10k/train/data.lmdb`
- Test: `.../is2re/all/val_id/data.lmdb`

Note that there is no validation/test split for this dataset and the provided validation split is used as the test split. Early stopping based on this set must be manually disabled when training.

The same applies for the dataset `PCQM4Mv2`, where the test sets are not publicly available and the validation set is used as a test set. Early stopping must be manually disabled in the code, and a fixed number of training epochs (we used 400) must be used.
  

# ESA introduction

The general template for starting ESA training looks like the following example:

```python -m esa.train --dataset <DS> --dataset-download-dir <DS_DIR> --dataset-one-hot --dataset-target-name None --lr 0.0001 --batch-size 128 --norm-type LN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 512 --apply-attention-on edge --use-mlps --mlp-hidden-size 512 --out-path <OUT_DIR> --wandb-project-name <WANDB_PROJ> --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type gated_mlp --use-bfloat16 --layer-types M M S P --pre-or-post post --hidden-dims 256 256 256 256 --num-heads 4 4 4 4 --mlp-dropout 0 --mlp-layers 2 --mlp-hidden-size 512```

Some arguments control general training parameters that are shared with other models, such as `--lr`, `--batch-size`, `--optimiser-weight-decay`, `--gradient-clip-val 0.5`. The setting `--monitor-loss-name` provides the name of the loss to track for early stopping (`val_loss/dataloader_idx_0` for validation loss, `train_loss` for the training loss).

The `--dataset-download-dir` argument is used to provide a path to the dataset. Datasets provided by PyTorch Geometric will be downloaded in this directory (and it will be created), while for other datasets such as DOCKSTRING the user must manually place the files provided in this code base to the desired path and pass it to this argument. **IMPORTANT**: provide a unique path for each dataset, e.g. /datasets/MNIST, and for each dataset + target combination for QM9 and DOCKSTRING, e.g. /datasets/QM9/homo, as the data loading code might create additional files and directories.

Similarly, `--out-path` is used as the output path that will store all the models outputs. This will be created if it does not exist.

`--wandb-project-name` can be used to specify a project name for Weights and Biases (you need to log in; the project will be created for you if it does not exist already).

Apart from the input path, two options control the dataset settings. `--dataset` is used to specify the name of the dataset, and must adhere to some conventions for certain datasets such as the heterophily ones (see below for examples). `--dataset-one-hot` produces one-hot encodings for node features, for example for some molecular datasets which use ChemProp featurisation or for datasets that use integer features. This is the default option for ESA. Some algorithms, such as Graphormer and TokenGT, do not support one-hot encodings as they use embedding layers internally. In these cases, the `--no-dataset-one-hot` must be used.

The option `--use-bfloat16` controls the use of mixed precision training with the `bfloat16` data type. It can be disabled using `--no-use-bfloat16` (but we used it for all experiments).

In terms of settings specific to ESA, the most important ones are:

-  `-graph-dim`, the hidden dimension of the graph representation after pooling node representations

-  `--norm-type`, controls the type of normalisation before/after attention. Current choices are `LN` (layer normalisation) and `BN` (batch normalisation).

-  `--apply-attention-on`, whether to use ESA by providing `edge`, or the simpler node-based alternative (NSA) by providing `node`

-  `--use-mlps`, whether to use MLPs after each attention block. Use `--no-use-mlps` to deactivate

-  `--hidden-dims`, the hidden dimensions for each attention block in ESA. Must match the number of layers defined in the `--layer-types` option (see below). Currently, the hidden dimensions do not have to be identical, but the output of one layer must be the same as the input to the next

-  `--num-heads`, the number of attention heads for the attention modules in ESA. Must adhere to the same conventions as `--hidden-dims` above

-  `--sab-dropout`, the dropout value to use for self-attention blocks (`S` in `--layer-types`)

-  `--mab-dropout`, the dropout value to use for masked self-attention blocks (`M` in `--layer-types`)

-  `--pma-dropout`, the dropout value to use for the pooling-by-multihead (PMA) attention block (`P` in `--layer-types`)

-  `--attn-residual-dropout`, the dropout value to use after a SAB/MAB block

-  `--pma-residual-dropout`, the dropout value to use after a PMA block

-  `--xformers-or-torch-attn`, the underlying efficient attention implementation. Currently only `xformers` and `torch` are implemented, but `xformers` seems to lead to better model performance and it is more efficient in terms of memory in some scenarios.

-  `--mlp-type`, the type MLP to use. Choices are `standard` (classic MLP) or `gated_mlp` (a gated MLP implementation)

-  `--mlp-layers`, the number of layers in the MLPs (inside the ESA model)

-  `--mlp-hidden-size`, the hidden size of the MLP, if enabled using `--use-mlps` (otherwise the setting has no effect)

-  `--mlp-dropout`, the dropout value to use within MLPs

-  `--use-mlp-ln`, whether to use layer normalisation inside MLPs (choices: `yes`, `no`)

-  `--layer-types`, specifies the number and order of layers. Choices are `S` for standard self-attention, `M` for masked self-attention, and `P` for the PMA decoder. `S` and `M` layers can be alternated in any order as desired. For graph-level tasks, there must be a single `P` layer specified. The `P` layer can be followed by `S` layers, but not by `M` layers. For node-level tasks, `P` must not be specified. The number of layers specified in this option must match the hidden dimensions and heads (see above)

-  `--pre-or-post`, whether to use pre-LN or post-LN layer ordering

-  `--pos-enc`, if specified, which type of positional encodings (PE) to use. By defeault, no PE are used. The choices for this argument are `RWSE`, `LapPE`, and `RWSE+LapPE` (for both)

Finally, `--seed` can be used to enable different initialisations depending on the integer seed. We use seeds in the range [0, 4] for the paper.

## Additional commands

For the datasets QM9 and DOCKSTRING, the user must specify the target/task using the command `--dataset-target-name`, for example `--dataset-target-name alpha` for QM9 or `--dataset-target-name PGR` for DOCKSTRING. The full list of available tasks is available in the `data_loading.py` file.

For regression datasets (for example QM9 and DOCKSTRING, but also others), the user must specify a regression function. The current options are the MAE and the MSE: `--regression-loss-fn mae` or `--regression-loss-fn mse`.

These options apply to ESA, GNNs, and Graphormer/TokenGT. Please check the instructions and the source code files for each to make sure that everything is setup correctly.


# Code changes for huggingface models
We have adapted Graphormer and TokenGT implementations from huggingface to work with node-level and 3D tasks. To work correctly, these require some changes in the `trainer.py` file that is part of the `transformers` installation. These are required for our code to work as intended, and only apply to the pip versions described below in **Requirements**.

## Node-level tasks
1. Replace `trainer.py` with the code provided in `hf_trainer_nodes.py`.
2. In the file `transformers/modeling_utils.py`, add the code:
    `model_to_save.config.update({"train_mask": "", "val_mask": "", "test_mask": ""})`
at line 2063 since saving the config does not support tensors. The surrounding code at this location should look like:

```
    # Save the config
    if is_main_process:
        if not _hf_peft_config_loaded:
            model_to_save.config.update({"train_mask": "", "val_mask": "", "test_mask": ""})
            model_to_save.config.save_pretrained(save_directory)
```

## 3D tasks
Make sure that the code that disables "unused" columns in `trainer.py` is disabled. An example is provided in `hf_trainer_unused_columns.py` (this can also be copied over `trainer.py`).

# Other graph-level tasks

## GNNs

The GNN code can be run using a command like the following:

```python -m gnn.train --dataset <DS> --dataset-download-dir <DS_DIR> --dataset-target-name None --dataset-one-hot --output-node-dim 128 --num-layers 4 --conv-type GCN --gnn-intermediate-dim 512 --batch-size 128 --lr 0.0001 --monitor-loss-name val_loss/dataloader_idx_0 --early-stopping-patience 30 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJ> --seed 0 --gradient-clip-val 0.5 --optimiser-weight-decay 1e-10 --train-regime gpu-bf16```

Note that in addition to the general settings covered above, there are a few options specific to GNNs.

There are 6 convolution variations available, controlled through `--conv-type <conv>`, where <conv> is in [`GCN`, `GIN`, `GAT`, `GATv2`, `PNA`, `GINDrop`].

GAT/GATv2 also take the additional arguments: `--gat-attn-heads` for the number of attention heads (usually set to 8) and `--gat-dropout` for the dropout used by this layer (usually set to 0.1).

Other GNN-specific options include:

-  `--output-node-dim`, the hidden dimension of the node

-  `--num-layers`, the number of convolutional/graph layers. We default to 4.

-  `--train-regime`, with the following options [`gpu-32`, `gpu-bf16`, `gpu-fp16`, `cpu`]

## Graphormer/TokenGT

The Graphormer/TokenGT code can be run using a command like the following:

**Graphormer**

```python -m graphormer_tokengt.train_graphormer_tokengt --dataset_name <DS> --dataset_download_dir <DS_DIR> --no-dataset_one_hot --dataset_target_name None --batch_size 128 --out_dir <OUT_DIR> --name <WANDB_RUN_NAME> --<WANDB_PROJ> --architecture graphormer --early_stop_patience 30 --embedding_dim 512 --hidden_size 512 --num_layers 8 --num_attention_heads 8 --gradient_clip_val 0.5 --optimiser_weight_decay 1e-10 --lr 0.0001 --seed 0 --bfloat16 yes```

**TokenGT**

```python -m graphormer_tokengt.train_graphormer_tokengt --dataset_name <DS> --dataset_download_dir <DS_DIR> --no-dataset_one_hot --dataset_target_name None --batch_size 128 --out_dir <OUT_DIR> --name <WANDB_RUN_NAME> --<WANDB_PROJ> --architecture tokengt --early_stop_patience 30 --embedding_dim 512 --hidden_size 512 --num_layers 8 --num_attention_heads 8 --gradient_clip_val 0.5 --optimiser_weight_decay 1e-10 --lr 0.0001 --seed 0 --bfloat16 yes```

The specific arguments for Graphormer and TokenGT are:

-  `--architecture`, which selects whether to use Graphormer (`graphormer`) or TokenGT (`tokengt`)
-  `--embedding_dim`, the embedding dimension used inside the model
-  `--hidden_size`, the hidden dimension used inside the model
-  `--num_layers `, the number of layers in the model
-  `--num_attention_heads`, the number of attention heads in the model
- 
Using mixed precision training can be turned on or off using `--bfloat16 yes` or `--bfloat16 no`.

Note that Graphormer and TokenGT are only available for tasks where the node/edge features are represented as integers. A list of compatible datasets in terms of integer features is provided in the file `data_loading.py`, in the list `DATASETS_WITH_INTEGER_FEATURES`.

## GraphGPS

GraphGPS is best run through configuration files (.yaml). We have adapted the GraphGPS code base and plugged in our data loading pipeline to ensure a fair comparison.

The notebook `helper/notebooks_graphgps/generate_scripts_graphgps_graph.ipynb` provides an automated approach to generating configuration files and runnable scripts for graph-level tasks using GraphGPS.

# Node-level tasks

## NSA

A typical command looks like:

```python -m esa.train --dataset <DS> --dataset-download-dir <DS_DIR> --dataset-one-hot --dataset-target-name None --lr 0.0001 --batch-size 128 --norm-type LN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 512 --apply-attention-on node --use-mlps --mlp-hidden-size 512 --out-path <OUT_DIR> --wandb-project-name <WANDB_PROJ> --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type gated_mlp --use-bfloat16 --layer-types M M S --pre-or-post post --hidden-dims 256 256 256 --num-heads 4 4 4 --mlp-dropout 0 --mlp-layers 2 --mlp-hidden-size 512```

Note that there are only 2 important changes compared to graph-level tasks:

1. We specify `--apply-attention-on node` since we don't currently support edge-based learning for node tasks.
2. The layer types do not include a PMA layer (`--layer-types M M S`). The number of hidden dimensions and heads is adjusted accordingly.

## GNNs

The command format is identical and there are no changes required.

```python -m gnn.train --dataset <DS> --dataset-download-dir <DS_DIR> --dataset-target-name None --dataset-one-hot --output-node-dim 128 --num-layers 4 --conv-type GCN --gnn-intermediate-dim 512 --batch-size 128 --lr 0.0001 --monitor-loss-name val_loss/dataloader_idx_0 --early-stopping-patience 30 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJ> --seed 0 --gradient-clip-val 0.5 --optimiser-weight-decay 1e-10 --train-regime gpu-bf16```

## Graphormer

The arguments are the same, but the appropriate script must be called.

```python -m graphormer_tokengt_nodes.train_graphormer_tokengt_nodes --dataset_name <DS> --dataset_download_dir <DS_DIR> --no-dataset_one_hot --dataset_target_name None --batch_size 128 --out_dir <OUT_DIR> --name <WANDB_RUN_NAME> --<WANDB_PROJ> --architecture graphormer --early_stop_patience 30 --embedding_dim 512 --hidden_size 512 --num_layers 8 --num_attention_heads 8 --gradient_cli_val 0.5 --optimiser_weight_decay 1e-10 --lr 0.0001 --seed 0 --bfloat16 yes```

Note that we are using `python -m graphormer_tokengt_nodes.train_graphormer_tokengt_nodes` instead of `python -m graphormer_tokengt.train_graphormer_tokengt`.

## TokenGT

The arguments are the same, but the appropriate script must be called.

```python -m graphormer_tokengt_nodes.train_graphormer_tokengt_nodes --dataset_name <DS> --dataset_download_dir <DS_DIR> --no-dataset_one_hot --dataset_target_name None --batch_size 128 --out_dir <OUT_DIR> --name <WANDB_RUN_NAME> --<WANDB_PROJ> --architecture tokengt --early_stop_patience 30 --embedding_dim 512 --hidden_size 512 --num_layers 8 --num_attention_heads 8 --gradient_clip_val 0.5 --optimiser-weight-decay 1e-10 --lr 0.0001 --seed 0 --bfloat16 yes```

Note that we are using `python -m graphormer_tokengt_nodes.train_graphormer_tokengt_nodes` instead of `python -m graphormer_tokengt.train_graphormer_tokengt`.

## GraphGPS

GraphGPS is best run through configuration files (.yaml). We have adapted the GraphGPS code base and plugged in our data loading pipeline to ensure a fair comparison. The adaptation to node-level tasks required further modifications to ensure compatibility with our data loading.

The notebook `helper/notebooks_graphgps/generate_scripts_graphgps_node.ipynb` provides an automated approach to generating configuration files and runnable scripts for node-level tasks using GraphGPS.

## Transfer learning (3D)

The transfer learning code is similar to the graph-level code, but with the following modifications/additions:

1. Support for modelling 3D atomic coordinates directly, without node and edge features.
2. The code must distinguish between low-fidelity/quality (LQ) and high-fidelity/quality (HQ) data
3. The code must distinguish between transductive and inductive scenarios.
4. The code supports fine-tuning a previously-trained LQ model on HQ data.

## ESA

### HQ (GW) only

In this case, we train exclusively on the HQ data (GW), as for any other graph-level benchmark:

```python -m transfer_learning.esa_3d.train --dataset-download-dir <DS_DIR> --dataset-target-name homo_gw --lr 0.0001 --batch-size 128 --norm-type LN --early-stopping-patience 10 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 512 --apply-attention-on edge --use-mlps --mlp-hidden-size 512 --out-path <OUT_DIR> --wandb-project-name <WANDB_PROJ> --hidden-dims 256 256 256 256 256 256 256 256 256 --num-heads 16 16 16 16 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type gated_mlp --use-bfloat16 --layer-types M M M M M M M M P --mlp-layers 2 --pma-residual-dropout 0 --use-mlp-ln yes --pre-or-post post --mlp-dropout 0 --attn-residual-dropout 0 --regression-loss-fn mae --transfer-learning-hq-or-lq hq```

Two important things to note:

1.  `--dataset-target-name` must be set to one of the HQ targets (GW): either `homo_gw` or `lumo_gw`.
2. We specify that we are using the HQ data splits using `--transfer-learning-hq-or-lq hq`.

### LQ (DFT) only (pre-training)

In this case, we train exclusively on the LQ data (DFT). However, this acts as pre-training and validation and test splits are not used. By default we train for 150 epochs.

```python -m transfer_learning.esa_3d.train --dataset-download-dir <DS_DIR> --dataset-target-name homo_dft --lr 0.0001 --batch-size 128 --norm-type LN --early-stopping-patience 10 --monitor-loss-name train_loss --graph-dim 512 --apply-attention-on edge --use-mlps --mlp-hidden-size 512 --out-path <OUT_DIR> --wandb-project-name <WANDB_PROJ> --hidden-dims 256 256 256 256 256 256 256 256 256 --num-heads 16 16 16 16 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type gated_mlp --use-bfloat16 --layer-types M M M M M M M M P --mlp-layers 2 --pma-residual-dropout 0 --use-mlp-ln yes --pre-or-post post --mlp-dropout 0 --attn-residual-dropout 0 --regression-loss-fn mae --transfer-learning-hq-or-lq lq --transfer-learning-inductive-or-transductive transductive```

Three important things to note:

1.  `--dataset-target-name` must be set to one of the LQ targets (DFT): either `homo_dft` or `lumo_dft`.
2. We specify that we are using the LQ data splits using `--transfer-learning-hq-or-lq lq`.
3. We must specify the pre-training scenario: `--transfer-learning-inductive-or-transductive`, with possible settings being `transductive` and `inductive`.

### Fine-tuning LQ to HQ (DFT to GW)

Here, we use a previously-trained LQ (DFT) model checkpoint to start fine-tuning:

```python -m transfer_learning.esa_3d.train --dataset-download-dir <DS_DIR> --dataset-target-name homo_gw --lr 0.0001 --batch-size 128 --norm-type LN --early-stopping-patience 10 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 512 --apply-attention-on edge --use-mlps --mlp-hidden-size 512 --out-path <OUT_DIR> --wandb-project-name <WANDB_PROJ> --hidden-dims 256 256 256 256 256 256 256 256 256 --num-heads 16 16 16 16 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type gated_mlp --use-bfloat16 --layer-types M M M M M M M M P --mlp-layers 2 --pma-residual-dropout 0 --use-mlp-ln yes --pre-or-post post --mlp-dropout 0 --attn-residual-dropout 0 --regression-loss-fn mae --transfer-learning-hq-or-lq hq --transfer-learning-inductive-or-transductive transductive --transfer-learning-retrain-lq-to-hq yes --ckpt-path <CKPT_PATH>```

Five important things to note:

1.  `--dataset-target-name` must be set to one of the HQ targets (GW): either `homo_gw` or `lumo_gw`.
2. We specify that we are using the HQ data splits using `--transfer-learning-hq-or-lq hq`.
3. We must specify the pre-training scenario: `--transfer-learning-inductive-or-transductive`, with possible settings being `transductive` and `inductive`.
4. We must enable fine-tuning through `--transfer-learning-retrain-lq-to-hq yes`.
5. We must specify a checkpoint to a LQ (DFT) model with `--ckpt-path <CKPT_PATH>`.

## GNN

The 3 scenarios and the arguments also apply to GNNs. The corresponding scripts must be called using `python -m transfer_learning.gnn_3d.train`.

## Graphormer

We have adapted the official Graphormer 3D implementation to use the same transfer learning interface as above. A typical command looks like:

```python -m transfer_learning.graphormer_3d.train --dataset-download-dir <DS_DIR> --dataset-target-name homo_gw --blocks 4 --layers 8 --embed-dim 32 --ffn-embed-dim 512 --attention-heads 16 --input-dropout 0 --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0 --num-kernel 32 --batch-size 128 --lr 0.0001 --monitor-loss-name val_loss/dataloader_idx_0 --gradient-clip-val 0.5 --optimiser-weight-decay 1e-10 --regression-loss-fn mae --early-stopping-patience 30 --train-regime gpu-bf16 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJ> --seed 0 --proj-dim 512 --transfer-learning-hq-or-lq hq --transfer-learning-inductive-or-transductive inductive --transfer-learning-retrain-lq-to-hq yes --ckpt-path <CKPT_PATH>```

Note that some training arguments are specific to Graphormer 3D. These are explained in the original repository.

## TokenGT

We have adapted the huggingface TokenGT implementation for 3D data and transfer learning. The interface is identical to graph-level TokenGT, with the addition of the transfer learning options:

```python -m transfer_learning.tokengt_3d.train --dataset_download_dir <DS_DIR> --dataset_target_name homo_gw --batch_size 128 --out_dir <OUT_DIR> --name <WANDB_RUN_NAME> --wandb_project_name <WANDB_PROJ> --early_stop_patience 30 --embedding_dim 512 --hidden_size 512 --num_layers 8 --num_attention_heads 8 --gradient-clip-val 0.5 --optimiser-weight-decay 1e-10 --lr 0.0001 --seed 0 --bfloat16 yes --regression_loss_fn mae --transfer_learning_hq_or_lq hq --transfer_learning_inductive_or_transductive inductive --transfer_learning_retrain_lq_to_hq --ckpt_path <CKPT_PATH>```


## GraphGPS

GraphGPS is best run through configuration files (.yaml). We have adapted the GraphGPS code base and plugged in our data loading pipeline to ensure a fair comparison. We have further adapted the original code to support the same transfer learning protocols as for all the other methods, with specific configuration file arguments.

The notebook `helper/notebooks_graphgps/generate_scripts_graphgps_transfer_learning.ipynb` provides an automated approach to generating configuration files and runnable scripts for transfer learning tasks using GraphGPS.


## OCP (3D)

We also support learning on the OCP benchmark for ESA only, illustrated here and in the paper through a 10K official subset. The interface is identical to ESA for graph-level tasks, by specifying `--dataset <DS>` (see below for an example command). The main problem is the dependency of OCP on an old version of `fairseq`, which is not officially compatible with recent versions of PyTorch, xformers, and Flash attention. A working environment configuration file for `fairseq` is provided below.

# Examples

These examples illustrate the format for various benchmarks. The parameters and settings are randomly chosen.

## Graph-level tasks

Example commands for different datasets.

### QM9

```python -m esa.train --dataset QM9 --dataset-download-dir <DS_DIR> --dataset-one-hot --dataset-target-name homo --lr 0.0001 --batch-size 128 --norm-type BN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on edge --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 256 256 256 --num-heads 16 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --layer-types M S M S M P --pre-or-post post --regression-loss-fn mae```

### DOCKSTRING

```python -m esa.train --dataset DOCKSTRING --dataset-download-dir <DS_DIR> --dataset-one-hot --dataset-target-name PGR --lr 0.0001 --batch-size 128 --norm-type BN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on edge --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 256 256 256 --num-heads 16 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --layer-types M S M S M P --pre-or-post post --regression-loss-fn mse```

### MNIST

```python -m esa.train --dataset MNIST --dataset-download-dir <DS_DIR> --dataset-one-hot --lr 0.0001 --batch-size 128 --norm-type BN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on edge --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 256 256 256 --num-heads 16 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --pre-or-post post --layer-types M S M S M P```

### Long-range graph benchmark (LRGB) peptides

```python -m esa.train --dataset lrgb-pept-struct --dataset-download-dir <DS_DIR> --dataset-one-hot --dataset-target-name None --lr 0.0001 --batch-size 128 --norm-type BN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on edge --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 256 256 256 --num-heads 16 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --layer-types M S M S M P --pre-or-post post --regression-loss-fn mae```

The two possible choices for LRGB datasets are: `--dataset lrgb-pept-struct` and `--dataset lrgb-pept-fn`.

### OCP

This assumes that the train and test data splits have been downloaded as instructed above.

```python -m esa.train --dataset ocp --dataset-download-dir None --dataset-one-hot --dataset-target-name None --lr 0.0001 --batch-size 8 --norm-type LN --early-stopping-patience 15 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on node --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 --num-heads 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --layer-types M S P --regression-loss-fn mae --ocp-num-kernels 128 --ocp-embed-dim 128 --ocp-cutoff-dist 2 --ocp-num-neigh 16```

Note the OCP-specific options `--ocp-num-kernels 128 --ocp-embed-dim 128 --ocp-cutoff-dist 2 --ocp-num-neigh 16`. These are common in architectures working on 3D atomic data.

OCP training has only been tested with NSA, i.e. `--apply-attention-on node`.

You must disable the import of `bnb` when using the `fairseq` environment and switch the optimizer to the default AdamW in PyTorch.

## Node-level tasks

## Cora

```python -m esa.train --dataset Cora --dataset-download-dir <DS_DIR> --dataset-one-hot --lr 0.0001 --batch-size 128 --norm-type BN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on node --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 256 256 --num-heads 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --layer-types M S M S M --pre-or-post post```

## Shortest paths (infected)

```python -m esa.train --dataset infected+15000 --dataset-download-dir <DS_DIR> --dataset-one-hot --lr 0.0001 --batch-size 128 --norm-type BN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on node --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 256 256 --num-heads 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --layer-types M S M S M --pre-or-post post```

Note that for the shortest path (infected) datasets, the convention is to specify the dataset name as `infected+<NUM_NODES>`: `--dataset infected+15000` or `--dataset infected+30000` and provide the path to the directory containing the infected datasets, e.g. `--dataset-download-dir datasets/infected/`.

## Heterophily datasets

```python -m esa.train --dataset hetero+squirrel_filtered --dataset-download-dir <DS_DIR> --dataset-one-hot --lr 0.0001 --batch-size 128 --norm-type BN --early-stopping-patience 30 --monitor-loss-name val_loss/dataloader_idx_0 --graph-dim 256 --apply-attention-on node --use-mlps --mlp-hidden-size 256 --out-path <OUT_PATH> --wandb-project-name <WANDB_PROJECT> --hidden-dims 256 256 256 256 256 --num-heads 16 16 16 16 16 --sab-dropout 0 --mab-dropout 0 --pma-dropout 0 --seed 0 --optimiser-weight-decay 1e-10 --gradient-clip-val 0.5 --xformers-or-torch-attn xformers --mlp-type standard --use-bfloat16 --layer-types M S M S M --pre-or-post post```

Note that the convention for heterophily datasets is to provide the dataset name in the format `--dataset hetero+<DS_NAME>` (emphasis on the mandatory `hetero+` part). The part after the plus sign can be any of the included datasets, i.e. `squirrel_filtered`, `chameleon_filtered`, `roman_empire`, `amazon_ratings`, `minesweeper`, `tolokers`. You must provide the path to the directory containing the hetero datasets, e.g. `--dataset-download-dir datasets/hetero/`.

# Time and memory tracking

The memory utilisation can easily be reported directly from PyTorch, at the end of training. After training a model, i.e. after the `trainer.fit()` call inside `train.py` (for ESA), add the following lines:

```
max_memory_allocated = torch.cuda.max_memory_allocated()
max_memory_reserved = torch.cuda.max_memory_reserved()

print('Max memory allocated = ', max_memory_allocated * 1e-9)
print('Max memory reserved = ', max_memory_reserved * 1e-9)

np.save(os.path.join(output_save_dir, 'max_memory_allocated.npy'), max_memory_allocated)
np.save(os.path.join(output_save_dir, 'max_memory_reserved.npy'), max_memory_reserved)
```

For timing, we can make use of the `Timer` callback from PyTorch Lightning. In the corresponding training file, follow this structure:

```
# Import
from pytorch_lightning.callbacks import Timer
... # Argument parsing and data loading

# Define timer
timer_callback = Timer()

# Add callback to training arguments
trainer_args = dict(
	callbacks=[timer_callback],
	... # Other training arguments
)
...

# Define Trainer and run
trainer = pl.Trainer(**trainer_args)
trainer.fit(...)

# Get and save timing data
time_elapsed_train = timer_callback.time_elapsed('train')
print('Elapsed train time = ', time_elapsed_train)
np.save(os.path.join(output_save_dir, 'train_time.npy'), time_elapsed_train)
```

For Graphormer/TokenGT, the following alternative can be used:

```
# After defining the Trainer in train_graphormer_tokengt.py
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
trainer.train()
end.record()

torch.cuda.synchronize()
time_elapsed_train = start.elapsed_time(end)

print('Elapsed train time = ', time_elapsed_train)
np.save(os.path.join(args.out_dir, 'train_time.npy'), time_elapsed_train)
```

# Requirements

The code requires the following libraries to be installed: PyTorch, PyTorch Lightning, PyTorch Geometric, xformers, flash_attn (including the `fused_dense_lib` subcomponents), transformers, datasets, accelerate, bitsandbytes, wandb, Cython, ogb, admin-torch, and the LRGB dependencies from the official repository if LRGB tasks are desired. The code requires an NVIDIA GPU to run, preferably Ampere-class or newer.  

An example conda environment file `helper/conda_envs/env.yaml` is provided as an example of the required packages and to help with installation.

Efficient attention implementations are improving rapidly and we recommend installing the latest versions. At the time of writing, these are:

1. Create conda environment:
```mamba create --name torch-esa python=3.11 -y```
```mamba activate torch-esa```

2. Install PyTorch (2.5.1)
```mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y```

3. Install PyTorch Geometric (2.6.1) and auxiliary packages
```mamba install pyg -c pyg -y```
```mamba install pytorch-scatter -c pyg -y```
```mamba install pytorch-sparse -c pyg -y```
```mamba install pytorch-cluster -c pyg -y```

4. Install xformers (v0.0.28.post3)
```pip install xformers --index-url https://download.pytorch.org/whl/cu121```

5. Install Flash attention (v2.7.0.post2)
  ```pip install flash-attn --no-build-isolation```

6. Install specific version of transformers from huggingface
  ```pip install transformers==4.35.0 datasets==2.14.6 accelerate==0.24.1```
  These are required in order to make sure that some of the node-level/3D task adaptations work as intended.

7. Install other requirements
  ```pip install pytorch_lightning pandas scikit-learn wandb rdkit bitsandbytes yacs admin_torch Cython ogb ```

9. GraphGPS might have additional dependencies such as `performer-pytorch` (best to check official repo).

## Fairseq

Training on OCP requires the package `fairseq` to be installed. Unfortunately, Graphormer 3D uses the first version of `fairseq`, not `fairseq2`. The binaries of the old `fairseq` are not compatible with the latest versions of Python, PyTorch, and CUDA. Thus, the packages must be manually built from source which takes significant time and effort. Two versions of a `fairseq` environment export from `conda` were attached in an attempt to help reproduce our environment. The file `helper/conda_envs/fairseq_env.yaml` corresponds to a simple export, while the file `helper/conda_envs/fairseq_from_history_env.yaml` used the additional option `--from-history`.

**IMPORTANT**: The `.yaml` files are anonymised and you must replace the placeholder `<NAME>` and `<PREFIX>` if you want to use them directly.

## Citation

If you use ESA in your work, a citation to the paper is appreciated:

```
@Article{Buterez2025,
	author={Buterez, David
	and Janet, Jon Paul
	and Oglic, Dino
	and Li{\`o}, Pietro},
	title={An end-to-end attention-based approach for learning on graphs},
	journal={Nature Communications},
	year={2025},
	month={Jun},
	day={05},
	volume={16},
	number={1},
	pages={5244},
	issn={2041-1723},
	doi={10.1038/s41467-025-60252-z},
	url={https://doi.org/10.1038/s41467-025-60252-z}
}
```
