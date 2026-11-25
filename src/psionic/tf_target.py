'''
Functions for creating TF-target gene matrix
Assuming we use cell type x peak matrix as input, the function will first generate bed files for each cell type.
For each TF, it will calculate the overlap of peaks between each peak and each TF-binding reference bed file, 
and identify the best-matched reference bed file for each celltype.
In imputation step, per TF, scATACseq peak sets overlapping with all best-matched reference bed files are retained. 
Then for each cell type, peaks not in the cell-type-specific best-matched reference dataset are set to 0.
In target step, per TF, the intersected peak sets are used to calculate regulatory potential scores (a gene x peak matrix). 
Then the dot product of imputed peak count matrix (cell type x peak) and the regulatory potential score matrix (gene x peak) 
will give us the target-gene matrix (gene x cell type) of one TF with summarized regulatory potential scores.
'''

import os
import sys
import scipy
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import subprocess
import random
import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED, as_completed
from .tf_target_helpers import load_config, monitor_runtime_memory, logger, process_tf


@monitor_runtime_memory
def create_TF_target_matrix(
    input_matrix_path,
    factors,
    species="human",
    remove_others=True,
    ref_baseline=500,
    decay="auto",
    model='simple',
    n_cores=8,
    corr_csv_template=None
):
    """
    Wrapper to create target gene regulatory potential matrices.
    Runs imputation for each TF and maps peak-by-celltype results to gene-by-celltype matrices.

    Parameters
    ----------
    input_matrix_path : str
        Path to input ATAC matrix (.h5, .h5ad, or 10x folder)
    project : str
        Project name or output folder prefix
    factors : list
        List of TF names to impute (e.g., ["CTCF", "SPI1", "REST"])
    species : str, default="hs"
        Species code ('hs' or 'mm')
    remove_others : bool
        Whether to zero out peaks not from the best-matched reference
    ref_baseline : int, default=500
        Minimum baseline peaks for reference filtering
    decay : int or "auto"
        Half-decay distance for RP calculation
    model : str, default="enhanced"
        RP model to use ("simple" or "enhanced")
    refgene_path : str
        Path to reference gene annotation BED file
    write_format : str, optional
        Format to save output ('h5ad' or 'mtx')
    n_cores : int
        Parallel cores
    """
    
    #1. load chip index for human or mouse TF
    if species == 'human':
       chip_index, CONFIG_PATH = load_config(species="human")
       refgene_path = os.path.join(CONFIG_PATH, 'GRCh38_refgenes.txt')
    elif species == 'mouse':
       chip_index, CONFIG_PATH = load_config(species="mouse")
       refgene_path = os.path.join(CONFIG_PATH, 'GRCm38_refgenes.txt')
    else:
        pass


    #2. load peak count matrix and QC

    # load once
    if input_matrix_path.endswith(".h5"):
        feature_matrix = sc.read_10x_h5(input_matrix_path, gex_only=False)
    elif input_matrix_path.endswith(".h5ad"):
        feature_matrix = sc.read_h5ad(input_matrix_path)
    else:
        feature_matrix = sc.read_10x_mtx(input_matrix_path)


    # 3. Parallel TF-level imputation
    results = {}

    # Run TFs concurrently
    logger.info(f"Running {len(factors)} TFs in parallel...")
    with ProcessPoolExecutor(max_workers=min(len(factors), n_cores)) as executor:
        futures = {executor.submit(process_tf, tf, feature_matrix, chip_index, ref_baseline,
                               remove_others, decay, model, refgene_path, n_cores, len(factors), corr_csv_template ): tf for tf in factors}
        for future in as_completed(futures):
            tf, gene_matrix = future.result()
            results[tf] = gene_matrix

    ##print_log("\n✅ All factors processed successfully.")
    logger.info("✅ All factors processed successfully.")
    return results


# Combine TF results into cell type specific TF-target gene matirx
def combine_tf_results(results, out_dir=None, save=True):
    """
    Combine TF-specific AnnData results into TF x Gene matrix for each cell type.

    Returns
    -------
    dict
        {celltype: AnnData} TF x Gene matrices
    """
    # Prepare output directory
    if save and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # Get all cell types (assumed same across TFs)
    example_adata = next(iter(results.values()))
    celltypes = example_adata.obs_names

    # Collect all genes from all TFs
    all_genes = sorted(
        set().union(*(adata.var_names for adata in results.values()))
    )

    # Create per-celltype TF x Gene DataFrames
    combined = {
        celltype: pd.DataFrame(index=results.keys(),
                               columns=all_genes,
                               dtype=float)
        for celltype in celltypes
    }

    # Fill in the matrices
    for tf, adata in results.items():
        for celltype in adata.obs_names:
            gene_values = pd.Series(
                adata[celltype, :].X.toarray().ravel(),
                index=adata.var_names
            )
            combined[celltype].loc[tf, gene_values.index] = gene_values

    # Convert each celltype matrix to AnnData
    celltype_adata_dict = {}
    for ct, df in combined.items():
        X = df.fillna(0).to_numpy()
        adata_ct = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=df.index),   # TFs
            var=pd.DataFrame(index=df.columns)  # Genes
        )
        celltype_adata_dict[ct] = adata_ct

        if save:
            fname = f"{ct}_TFxGene.h5ad"
            if out_dir is not None:
                fname = os.path.join(out_dir, fname)
            adata_ct.write_h5ad(fname)

    return celltype_adata_dict



