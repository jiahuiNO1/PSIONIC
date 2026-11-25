
import os
import ruamel.yaml
import time
import psutil
import sys
import gzip
import platform
import subprocess
import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
from multiprocessing import Pool
#import pybedtools
import shutil
import pyranges as pr
import glob
import logging
from collections import defaultdict
import functools
import gc



# Library logger: silent by default 
# Call psionic.set_log_file(...) to write log messages to a file.
logger = logging.getLogger("psionic")
logger.addHandler(logging.NullHandler())


def set_log_file(log_file="tf_imputation_full.log", mode="w"):
    """
    Write PSIONIC log messages to a file.

    Parameters
    ----------
    log_file : str
        Path of the log file. Default: "tf_imputation_full.log".
    mode : str
        "w" to overwrite an existing file (default), "a" to append.

    Returns
    -------
    logging.Logger
        The PSIONIC logger.
    """
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # keep messages out of Jupyter output

    # Remove handlers from earlier calls 
    for h in list(logger.handlers):
        if not isinstance(h, logging.NullHandler):
            logger.removeHandler(h)
            h.close()

    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Backward-compatible alias for the previous function name
setup_logger = set_log_file


def execute_info(start_info=None, end_info=None):
    def call(func):
        def wrapper(*args, **kwargs):
            if start_info:
                logger.info(start_info)
            ret = func(*args, **kwargs)
            if end_info:
                logger.info(end_info)
            return ret
        return wrapper
    return call

def safe_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(species="human"):
    """
    Load and update the YAML config file with the TF index path.

    Parameters
    ----------
    species : str, optional
        Species type. Accepts 'human' or 'mouse' (default: 'human').

    Returns
    -------
    chip_index : str
        The absolute path to the selected TF index from the config file.
    CONFIG_PATH : str
        The path to the configuration directory.
    """

    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False

    # Locate config paths
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()  # fallback for notebooks

    #CONFIG_PATH = os.path.abspath(os.path.join(base_dir, '..', 'conf'))
    CONFIG_PATH = os.path.abspath(os.path.join(base_dir, 'conf'))
    CONFIG_YML_PATH = os.path.join(CONFIG_PATH, 'config.yml')

    # Load existing config
    with open(CONFIG_YML_PATH, 'r') as config_file:
        CONFIG = yaml.load(config_file)

    # Determine species-specific TF directory
    if species == "human":
        tf_dir = os.path.join(CONFIG_PATH, 'TR_human/raw_beds')
        CONFIG["index"]["human_tf_index"] = os.path.abspath(tf_dir)
        chip_index = CONFIG["index"]["human_tf_index"]
    elif species == "mouse":
        tf_dir = os.path.join(CONFIG_PATH, 'TR_mouse/raw_beds')
        CONFIG["index"]["mouse_tf_index"] = os.path.abspath(tf_dir)
        chip_index = CONFIG["index"]["mouse_tf_index"]
    else:
        raise ValueError("Invalid hs value. Use 'human' or 'mouse'.")

    # Save updated config
    with open(CONFIG_YML_PATH, 'w') as config_file:
        yaml.dump(CONFIG, config_file)

    #print(f'✅ Config updated successfully for {species} TF index!')
    #logger.info(f'✅ Config updated successfully for {species} TF index!')

    return chip_index, CONFIG_PATH




def generate_peak_list(cells, input_mat, peak_confidence=1):
    cell_above_cutoff_index = sc.pp.filter_genes(
        input_mat[cells, :], min_cells=peak_confidence, inplace=False)[0]
    peaks = input_mat.var_names[cell_above_cutoff_index].to_list()
    return peaks

def generate_beds(file_path, cells, input_mat, peak_confidence=1):
    """
    Generates a compressed BED file for a cell type. 
    """
    peaks = generate_peak_list(cells, input_mat, peak_confidence)
    
    if len(peaks) == 0:
        logger.info(f'Warning: No peaks in {os.path.basename(file_path)}, skip generation.')
        return
    
    # Split "chr_start_end" string into 3 columns
    peaks_df = pd.DataFrame([p.rsplit("_", 2) for p in peaks])
    
    # Check platform for optimized sorting/compression
    system = platform.system()
    bgzip_path = shutil.which("bgzip")

    if system in ["Linux", "Darwin"] and bgzip_path is not None:
        # Use system tools for speed
        peaks_df.to_csv(file_path, sep="\t", header=None, index=None)
        cmd = (
            f"sort --buffer-size 2G -k1,1 -k2,2n -k3,3n {file_path} | "
            f"{bgzip_path} -c > {file_path}.gz && rm {file_path}"
        )
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Python-based fallback for Windows or environments without bgzip
        peaks_df = peaks_df.sort_values(by=[0, 1, 2])
        with gzip.open(f"{file_path}.gz", 'wt') as f:
            peaks_df.to_csv(f, sep='\t', header=False, index=False)
    
    return

@execute_info('Start generating cells beds ...', 'Finished generating cells beds!')
def generate_beds_by_matrix(cell_feature_adata, beds_path, n_cores):
    """
    Iterates through the AnnData object to generate BED files for all cell types.
    """
    safe_makedirs(beds_path)

    executor = ThreadPoolExecutor(max_workers=n_cores)
    all_task = []

    for cell in cell_feature_adata.obs.index:
        bed_gz = os.path.join(beds_path, f"{cell}.bed.gz")

        # Skip if file already exists (useful for resuming runs)
        if os.path.exists(bed_gz):
            continue

        all_task.append(
            executor.submit(
                generate_beds,
                os.path.join(beds_path, f"{cell}.bed"),
                cell,
                cell_feature_adata
            )
        )

    wait(all_task, return_when=ALL_COMPLETED)


def search_ref_factor(bed_path, result_path, index_path, factor):
    """
    Output columns:
    file, file_size, overlaps
    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Load query BED file
    query = pr.read_bed(bed_path)

    # Get reference BED files for the given factor
    ref_files = glob.glob(os.path.join(index_path, f"{factor}_*.bed.gz"))
    if not ref_files:
        raise FileNotFoundError(f"No reference BED files found for factor '{factor}' in {index_path}")

    results = []

    for ref_file in ref_files:
        reference = pr.read_bed(ref_file)
        n_ref = len(reference)

        # Count overlaps
        overlaps = len(query.join(reference))
        results.append([os.path.basename(ref_file), n_ref, overlaps])

    # Save results
    df = pd.DataFrame(results, columns=["file", "file_size", "overlaps"])
    with open(result_path, "w") as f:
        header_line = "#" + "\t".join(df.columns.tolist()) + "\n"
        f.write(header_line)
        df.to_csv(f, sep="\t", index=False, header=False)
    #print(f"✅ PyRanges overlap search completed for {factor}: saved to {result_path}")
    #logger.info(f"✅ PyRanges overlap search completed for {factor}: saved to {result_path}")


def search_ref_factor_batch(bed_folder, result_folder, index_path, factor, n_cores=8, tp=''):
    ##print_log(f'Start searching beds from {tp} index ...')
    #logger.info(f'Start searching beds from {tp} index ...')
    safe_makedirs(result_folder)
    beds = os.listdir(bed_folder)
    args = []
    for bed in beds:
        barcodes = bed[:-7]  # remove suffix '.bed.gz'
        args.append((os.path.join(bed_folder, bed),
                     os.path.join(result_folder, barcodes + '.txt'),
                     index_path,
                     factor))
    with Pool(n_cores) as p:
        p.starmap(search_ref_factor, args)
    ##print_log(f'Finished searching beds from {tp} index ...')
    #logger.info(f'Finished searching beds from {tp} index ...')

def read_search_result(files):
    for idx, file_path in enumerate(files):
        result_name = os.path.basename(file_path)
        cell_bc = result_name[:-4]  # remove suffix '.txt'
        dtframe = pd.read_csv(file_path, sep="\t", index_col=0, comment='#', header=None)
        read_col = 2  # 1 file_size 2 overlaps 3 odds_ratio 4 fishers_two_tail 5 fishers_left_tail 6 fishers_right_tail 7 combo_score
        if idx == 0:
            dtframe = dtframe.loc[:, [read_col]].copy()
            dataset_cell_score_df = dtframe.rename(columns={read_col: cell_bc}).copy()
        else:
            dataset_cell_score_df[cell_bc] = dtframe.loc[:, read_col]
    dataset_cell_score_df.index = [n.rsplit('/', 1)[0][:-7] for n in dataset_cell_score_df.index]  # remove suffix '.bed.gz'
    return dataset_cell_score_df


def read_search_result_batch(path, n_cores=8, tp=''):
    ##print_log(f"Reading searching results, using {n_cores} cores...")
    file_list = os.listdir(path)
    result_split = np.array_split(file_list, n_cores)
    args = [[os.path.join(path, j) for j in list_chunk] for list_chunk in result_split]
    with Pool(n_cores) as p:
        result = p.map(read_search_result, args)
    dataset_cell_score_df = pd.concat([i for i in result], axis=1)
    ##print_log(f"Finished reading {tp} index search result!")
    return dataset_cell_score_df

@execute_info('Getting the best reference for each cell.')
def get_factor_source(table):
    ret_table = table.copy()
    # map factor by id "_"
    factor_index_list = []
    for i in ret_table.index:
        factor_name = i.split("_")
        factor_index_list.append(factor_name[0])
    ret_table.loc[:, "Factor"] = factor_index_list
    max_index = ret_table.groupby("Factor").idxmax()
    return max_index


def parallel_join_by_chr(data_bed, chip_bed, n_cores=8):
    """
    Perform PyRanges join() in parallel by chromosome.
    Returns a single PyRanges object containing all overlaps.
    """
    # Get all chromosomes present in the input data
    chrs = list(data_bed.chromosomes)
    results = []

    def join_chr(chr_):
        """Join peaks for one chromosome."""
        return data_bed[data_bed.Chromosome == chr_].join(chip_bed)

    # Run joins in parallel
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(join_chr, chr_): chr_ for chr_ in chrs}
        for future in as_completed(futures):
            results.append(future.result())

    # Combine all chromosome-level overlaps
    return pr.concat(results)

def _process_single_tf(tf, factor_source, chip_cell_peak_df, peaks_pr, ref_path):

    # Get cell types for this TF
    cellbc = factor_source.columns[factor_source.iloc[0, :] == tf]

    # Load reference peaks
    tmp_dataset_bed = pr.read_bed(os.path.join(ref_path,  tf + '.bed.gz'))

    # Find overlaps
    overlap = peaks_pr.join(tmp_dataset_bed)
    overlap_ids = set(
        overlap.df["Chromosome"].astype(str) + "_" +
        overlap.df["Start"].astype(str) + "_" +
        overlap.df["End"].astype(str)
    )

    # Identify peaks not overlapping
    non_overlap_mask = ~chip_cell_peak_df.columns.isin(overlap_ids)
    chip_cell_peak_df.loc[cellbc, non_overlap_mask] = 0

    return chip_cell_peak_df.loc[cellbc, :]

def remove_others_source_batch(factor_source, chip_cell_peak_df, peaks_pr, ref_path, n_cores=1):

    unique_tfs = factor_source.iloc[0, :].unique()

    if n_cores > 1 and len(unique_tfs) > 1:
        # Prepare argument tuples
        args_list = [
            (tf, factor_source, chip_cell_peak_df.copy(), peaks_pr, ref_path)
            for tf in unique_tfs
        ]

        # Run in parallel using Pool.starmap
        with Pool(processes=n_cores) as p:
            results = p.starmap(_process_single_tf, args_list)

        # Merge back
        for res in results:
            if res is not None:
                chip_cell_peak_df.loc[res.index, res.columns] = res
    else:
        # Sequential processing
        for tf in unique_tfs:
            res = _process_single_tf(tf, factor_source, chip_cell_peak_df, peaks_pr, ref_path)
            if res is not None:
                chip_cell_peak_df.loc[res.index, res.columns] = res

    return chip_cell_peak_df


def impute(input_mat_adata, impute_factor, ref_path,  path='SCRIP/imputation',  ref_baseline=500, remove_others_source=False, n_cores=8):
    '''
    impute tf binding data from ATAC data
    '''

    safe_makedirs(path)
    #print(input_mat_adata.X.shape)

    ##print_log('Generating beds...')
    #logger.info('Generating beds...')
    generate_beds_by_matrix(input_mat_adata, f'{path}/imputed_beds/',  n_cores)

    ##print_log('Calculating score...')
    #logger.info('Calculating score...')
    search_ref_factor_batch(f'{path}/imputed_beds/', f'{path}/{impute_factor}/imputed_results_{impute_factor}/', ref_path, impute_factor, n_cores)
    factor_enrich = read_search_result_batch(f'{path}/{impute_factor}/imputed_results_{impute_factor}/', n_cores)

    peaks_number = pd.read_csv(os.path.join(ref_path, 'peaks_number.txt'), sep='\t', header=None, index_col=0)
    peaks_number_factor = peaks_number.loc[[i for i in peaks_number.index if i.startswith(f'{impute_factor}_')], :].copy()
    peaks_number_baseline_index = peaks_number_factor.index[peaks_number_factor[1] > ref_baseline]
    # peaks_number_factor = peaks_number_factor.loc[peaks_number_baseline_index, :]

    factor_enrich = factor_enrich.loc[peaks_number_baseline_index, :].copy()
    factor_score = (factor_enrich.T/peaks_number_factor.loc[factor_enrich.index, 1]).T

    factor_source = get_factor_source(factor_score)
    #store_to_pickle(factor_source, f'{path}/{impute_factor}_dataset_source.pk')

    chip_bed_list = [pr.read_bed(os.path.join(ref_path, i + '.bed.gz')) 
                 for i in factor_source.iloc[0, :].unique()]
    if len(chip_bed_list) == 1:
        chip_bed = chip_bed_list[0]
    else:
        chip_bed = pr.concat(chip_bed_list)

    # Convert input matrix peaks to PyRanges
    logger.info('Convert input matrix peaks to PyRanges')
    peak_records = [p.rsplit('_', 2) for p in input_mat_adata.var_names]
    peak_df = pd.DataFrame(peak_records, columns=["Chromosome", "Start", "End"])
    peak_df["Start"] = peak_df["Start"].astype(int)
    peak_df["End"] = peak_df["End"].astype(int)

    data_bed = pr.PyRanges(peak_df)

    # Find overlaps
    logger.info('Find overlaps')
    #intersected = data_bed.join(chip_bed)
    intersected = parallel_join_by_chr(data_bed, chip_bed, n_cores=n_cores)

    # Keep unique peaks only 
    intersected = pr.PyRanges(intersected.df.drop_duplicates(["Chromosome", "Start", "End"]))

    # Convert back to the same format as your original imputed_chip_peak
    imputed_chip_peak = (
        intersected.df["Chromosome"].astype(str) + "_" +
        intersected.df["Start"].astype(str) + "_" +
        intersected.df["End"].astype(str)
    ).tolist()

    chip_cell_peak = input_mat_adata[:, imputed_chip_peak].copy()
    chip_cell_peak_df = chip_cell_peak.to_df()
    
    peak_records = [p.rsplit('_', 2) for p in imputed_chip_peak]
    peak_df = pd.DataFrame(peak_records, columns=["Chromosome", "Start", "End"])
    peak_df["Start"] = peak_df["Start"].astype(int)
    peak_df["End"] = peak_df["End"].astype(int)
    peaks_pr = pr.PyRanges(peak_df)

    if remove_others_source:
        logger.info("Removing peaks not overlapping reference TF beds...")
        chip_cell_peak_df = remove_others_source_batch(
            factor_source=factor_source,
            chip_cell_peak_df=chip_cell_peak_df,
            peaks_pr=peaks_pr,
            ref_path=ref_path,
            n_cores=n_cores
        )

    chip_cell_peak = sc.AnnData(chip_cell_peak_df)
    chip_cell_peak.X = scipy.sparse.csr.csr_matrix(chip_cell_peak.X)
    
    ##print_log('Finished!')
    logger.info('Finished impute!')
    return chip_cell_peak



def propCalc(genes_list, peaks_list):
    PEAK = 0
    GENE = 1

    w = genes_list + peaks_list
    D = {}
    A = {}
    w.sort()
    for elem in w:
        if elem[2] == PEAK:
            A[elem[-1]] = [elem[0], elem[1]]
            D[elem[-1]] = float("inf")
        else:
            for peak in list(A.keys()):
                p_elem = A[peak]
                if p_elem[0] == elem[0]:
                    D[peak] = elem[1] - p_elem[1]
                    A.pop(peak)
                else:
                    A.pop(peak)

    w.reverse()
    for elem in w:
        if elem[2] == PEAK:
            A[elem[-1]] = [elem[0], elem[1]]
        else:
            for peak in list(A.keys()):
                p_elem = A[peak]
                if p_elem[0] == elem[0]:
                    D[peak] = min(p_elem[1] - elem[1], D[peak])
                    A.pop(peak)
                else:
                    A.pop(peak)
    D_values = D.values()
    D_len = float(len(D_values))
    peak_within_1k = [i for i in D.values() if i <= 1000]
    prop = len(peak_within_1k) / D_len

    if prop >= 0.2:
        logger.info(
            "%.2f%% of peaks are located in the promotor region. "
            "This is more than the 20%% promoter-type threshold, so the decay distance is set to 1.0kb, "
            "appropriate for promotor-type analysis. The half decay distance can be specified in the parameters." % (100*prop)
        )
    else:
        logger.info(
            "%.2f%% of peaks are located in the promotor region. "
            "This is less than the 20%% promoter-type threshold, so the decay distance is set to 10.0kb, "
            "appropriate for enhancer-type analysis. The half decay distance can be specified in the parameters." % (prop*100)
        )
    return prop



def geneInfoSimple(gene_bed):
    genes_info = []
    genes_list = []
    fhd = open(gene_bed, 'rt')
    fhd.readline()  # skip the first line. In our current gene txt file, there is no '#' in the first line. We need to, perhaps, use the 'ExtractGeneInfo' function.
    for line in fhd:
        line = line.strip().split('\t')
        if not line[0].startswith('#'):
            if line[3] == "+":
                genes_info.append((line[2].replace('chr', ''), int(line[4]), 1, "%s@%s@%s" % (line[12], line[2], line[4])))
            else:
                genes_info.append((line[2].replace('chr', ''), int(line[5]), 1, "%s@%s@%s" % (line[12], line[2], line[5])))
                # gene_info [chrom, tss, 1, gene_unique]
    fhd.close()
    genes_info = list(set(genes_info))
    for igene in range(len(genes_info)):
        tmp_gene = list(genes_info[igene])
        genes_list.append(tmp_gene[3])
        tmp_gene[3] = igene
        genes_info[igene] = tmp_gene
    return genes_info, genes_list


def ExtractGeneInfo(gene_bed):
    """Extract gene information from gene bed file."""

    bed = pd.read_csv(gene_bed, sep="\t", header=0, index_col=False)
    bed['transcript'] = [x.strip().split(".")[0] for x in bed['name'].tolist()]
    bed['tss'] = bed.apply(lambda x: x['txStart'] if x['strand'] == '+' else x['txEnd'], axis=1)
    ### adjacent P+GB
    bed["start"] = bed.apply(lambda x: x['txStart']-2000 if x['strand'] == '+' else x['txStart'], axis=1)
    bed["end"] = bed.apply(lambda x: x['txEnd']+2000 if x['strand'] == '-' else x['txEnd'], axis=1)
    bed['promoter'] = bed.apply(lambda x: tuple([x['tss']-2000, x['tss']+2000]), axis=1)
    bed['exons'] = bed.apply(lambda x: tuple([(int(i), int(j)) for i, j in zip(x['exonStarts'].strip(',').split(','), x['exonEnds'].strip(',').split(','))]), axis=1)
    ### exon length
    bed['length'] = bed.apply(lambda x: sum(list(map(lambda i: (i[1]-i[0])/1000.0, x['exons']))), axis=1)
    bed['uid'] = bed.apply(lambda x: "%s@%s@%s" % (x['name2'], x['start'], x['end']), axis=1)
    bed = bed.drop_duplicates(subset='uid', keep="first")

    genes_info = []
    genes_list = []
    for irow, x in bed.iterrows():
        genes_info.append([x['chrom'], x['start'], x['end'], x['tss'], x['promoter'], x['exons'], x['length'], 1, x['uid']])
        genes_list.append(x['uid'])
    ### [chrom_0, start_1, end_2, tss_3, promoter_4, exons_5, length_6, 1_7, uid_8]
    return genes_info, genes_list



def RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, decay):
    """Multiple processing function to calculate regulation potential."""

    def Sg(x): return 2**(-x)
    def checkInclude(x, y): return all([x >= y[0], x <= y[1]])
    gene_distance = 15 * decay
    genes_peaks_score_array = scipy.sparse.dok_matrix((len(genes_info_full), len(peaks_info)), dtype=np.float64)
    peaks_info_inbody = []
    peaks_info_outbody = []

    w = genes_info_full + peaks_info
    A = {}

    w.sort()
#     print(w[:100])
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                ### NOTE: main change here
                ### if peak center in the gene area
                if all([g[0] == elem[0], elem[1] >= g[1], elem[1] <= g[2]]):
                    ### if peak center in the exons
                    if any(list(map(checkInclude, [elem[1]]*len(g[5]), list(g[5])))):
                        genes_peaks_score_array[gene_name, elem[-1]] = 1.0 / g[-4]
                        peaks_info_inbody.append(elem)
                    ### if peak cencer in the promoter
                    elif checkInclude(elem[1], g[4]):
                        tmp_distance = abs(elem[1]-g[3])
                        genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                        peaks_info_inbody.append(elem)
                    ### intron regions
                    else:
                        continue
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    ### remove genes in promoters and exons
    peaks_info_set = [tuple(i) for i in peaks_info]
    peaks_info_inbody_set = [tuple(i) for i in peaks_info_inbody]
    peaks_info_outbody_set = list(set(peaks_info_set)-set(peaks_info_inbody_set))
    peaks_info_outbody = [list(i) for i in peaks_info_outbody_set]

    print("peaks number: ", len(peaks_info_set))
    print("peaks number in gene promoters and exons: ", len(set(peaks_info_inbody_set)))
    print("peaks number out gene promoters and exons:", len(peaks_info_outbody_set))

    w = genes_info_tss + peaks_info_outbody
    A = {}

    w.sort()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = elem[1] - g[1]
                if all([g[0] == elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    w.reverse()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = g[1] - elem[1]
                if all([g[0] == elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    return genes_peaks_score_array


def RP_Simple(peaks_info, genes_info, decay):
    """Multiple processing function to calculate regulation potential."""

    Sg = lambda x: 2**(-x)
    gene_distance = 15 * decay
    genes_peaks_score_array = scipy.sparse.dok_matrix((len(genes_info), len(peaks_info)), dtype=np.float64)

    w = genes_info + peaks_info

    A = {}

    w.sort()
    for elem in w:
        if elem[2] == 1:
            A[elem[-1]] = [elem[0], elem[1]]
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = abs(elem[1] - g[1])
                if (g[0] != elem[0]) or (tmp_distance > gene_distance):
                    dlist.append(gene_name)
                else:
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
            for gene_name in dlist:
                del A[gene_name]

    w.reverse()
    for elem in w:
        if elem[2] == 1:
            A[elem[-1]] = [elem[0], elem[1]]
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = abs(g[1] - elem[1])
                if (g[0] != elem[0]) or (tmp_distance > gene_distance):
                    dlist.append(gene_name)
                else:
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
            for gene_name in dlist:
                del A[gene_name]

    return genes_peaks_score_array






@functools.lru_cache(maxsize=6)
def load_retained_string_pairs(csv_path, wm_has_chr=True):
    """
    TF-INDEPENDENT: read the correlation CSV once per cell type, in chunks, and
    cache the raw (gene, peak) string arrays. Args are str/bool only, so
    lru_cache works. Reused across all TFs in this worker.
    """
    genes_all, peaks_all = [], []
    chunksize = 2_000_000
    for chunk in pd.read_csv(csv_path, usecols=["gene", "peak"], chunksize=chunksize):
        genes = chunk["gene"].astype(str)
        peaks = chunk["peak"].astype(str)

        csv_has_chr = peaks.str.startswith("chr").any()
        if csv_has_chr and not wm_has_chr:
            peaks = peaks.str.replace(r"^chr", "", regex=True)
        elif wm_has_chr and not csv_has_chr:
            peaks = "chr" + peaks

        genes_all.append(genes.to_numpy())
        peaks_all.append(peaks.to_numpy())

    if not genes_all:
        return (np.array([], dtype=object), np.array([], dtype=object))
    return (np.concatenate(genes_all), np.concatenate(peaks_all))


def pack_pairs_for_tf(genes_arr, peaks_arr, symbol_to_rows, peak_to_col):
    if len(genes_arr) == 0:
        return np.array([], dtype=np.int64)
    packed_all = []
    # map peaks once
    pcol = pd.Series(peaks_arr).map(peak_to_col)
    valid_p = pcol.notna()
    genes_v = pd.Series(genes_arr)[valid_p].to_numpy()
    cols_v = pcol[valid_p].astype(np.int64).to_numpy()
    for gene, col in zip(genes_v, cols_v):
        rows = symbol_to_rows.get(gene)
        if rows:
            for r in rows:                # expand to all transcript rows
                packed_all.append((np.int64(r) << 32) | np.int64(col))
    if not packed_all:
        return np.array([], dtype=np.int64)
    return np.unique(np.array(packed_all, dtype=np.int64))


def prune_gene_peak_matrix(weight_coo, retained_packed_arr, shape, dtype,
                           verbose=False, tag=""):
    """
    Prune the COO weight matrix using sorted int64 packed keys (row<<32 | col).
    """
    if weight_coo.nnz == 0 or len(retained_packed_arr) == 0:
        if verbose:
            print(f"prune[{tag}]: kept 0/{weight_coo.nnz} (empty retained set)")
        return sp.csr_matrix(shape, dtype=dtype)

    matrix_rows = weight_coo.row.astype(np.int64)
    matrix_cols = weight_coo.col.astype(np.int64)
    matrix_packed_keys = (matrix_rows << 32) | matrix_cols

    idx = np.searchsorted(retained_packed_arr, matrix_packed_keys)
    idx[idx == len(retained_packed_arr)] = len(retained_packed_arr) - 1
    mask = (retained_packed_arr[idx] == matrix_packed_keys)

    if verbose:
        print(f"prune[{tag}]: kept {int(mask.sum())}/{weight_coo.nnz} "
              f"({mask.mean():.1%}); {len(retained_packed_arr)} pairs in CSV")

    if not mask.any():
        return sp.csr_matrix(shape, dtype=dtype)

    return sp.csr_matrix(
        (weight_coo.data[mask], (weight_coo.row[mask], weight_coo.col[mask])),
        shape=shape, dtype=dtype,
    )


def count_to_gene_by_RP(input_adata, decay=10000, model='simple', refgene_path='',
                        corr_csv_template=None):
    """
    corr_csv_template : str or None
        Format string with {ct} placeholder for the per-cell-type retained
        gene-peak CSV. When None, no pruning is applied.
    """
    cells_list = input_adata.obs.index.tolist()
    peaks_list = input_adata.var.index.tolist()

    genes_info, genes_list = geneInfoSimple(refgene_path)
    peaks_info = []
    for ipeak, peak in enumerate(peaks_list):
        peaks_tmp = peak.rsplit("_", maxsplit=2)
        peaks_info.append([peaks_tmp[0][3:], (int(peaks_tmp[1]) + int(peaks_tmp[2])) / 2.0, 0, ipeak])

    if decay == 'auto':
        prop = propCalc(genes_info, peaks_info)
        decay = 1000 if prop >= 0.2 else 10000

    if model == 'simple':
        genes_peaks_score_dok = RP_Simple(peaks_info, genes_info, decay)
    elif model == 'enhanced':
        genes_info, genes_list = ExtractGeneInfo(refgene_path)
        genes_info_tss, genes_info_full = [], []
        for igene in range(len(genes_info)):
            tmp_gene = genes_info[igene]
            genes_info_full.append(tmp_gene + [igene])
            genes_info_tss.append([tmp_gene[0], tmp_gene[3], tmp_gene[1], tmp_gene[2]] + tmp_gene[4:] + [igene])
        peaks_info = []
        for ipeak, peak in enumerate(peaks_list):
            peaks_tmp = peak.rsplit("_", maxsplit=2)
            peaks_info.append([peaks_tmp[0], (int(peaks_tmp[1]) + int(peaks_tmp[2])) / 2.0,
                               int(peaks_tmp[1]), int(peaks_tmp[2]), 0, peak, ipeak])
        genes_peaks_score_dok = RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, decay)
    else:
        sys.exit(100)

    base_weight = genes_peaks_score_dok.tocsr()
    base_dtype = base_weight.dtype
    shape = base_weight.shape

    # depth-normalize accessibility per cell type (CPM-style, no log)
    adata_norm = input_adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    X_norm = scipy.sparse.csr_matrix(adata_norm.X)        # N_ct x Q
    n_ct = X_norm.shape[0]
    G_t = shape[0]

    # ---- precompute ONCE per TF (not per cell type) ----
    weight_coo = genes_peaks_score_dok.tocoo()
    row_symbol_arr = np.array([g.split("@")[0] for g in genes_list])    
    peaks_arr = np.array(peaks_list)
    wm_has_chr = bool(len(peaks_list)) and str(peaks_list[0]).startswith("chr")

    
    symbol_to_rows = defaultdict(list)
    for i, s in enumerate(row_symbol_arr):
        symbol_to_rows[s].append(i)          # ALL transcript rows per symbol
    
    peak_to_col = {peak: i for i, peak in enumerate(peaks_arr)}

    # --- per cell type: mask weights, then dot that cell type's column ---
    cols = []
    for c, ct in enumerate(cells_list):
        if corr_csv_template is not None:
            genes_csv, peaks_csv = load_retained_string_pairs(
                corr_csv_template.format(ct=ct), wm_has_chr)
            retained_packed = pack_pairs_for_tf(
                genes_csv, peaks_csv, symbol_to_rows, peak_to_col)
            W_c = prune_gene_peak_matrix(
                weight_coo, retained_packed, shape, base_dtype, verbose=True, tag=ct)
        else:
            W_c = base_weight

        col = W_c.dot(X_norm[c, :].T)                     # G_t x 1 sparse
        cols.append(np.asarray(col.todense()).ravel())

    genes_cells_score = np.column_stack(cols)             # G_t x N_ct dense

    # --- per-column best-transcript collapse to symbols ---
    symbols = row_symbol_arr
    gene_symbol = sorted(set(symbols.tolist()))
    sym_to_idx = {s: k for k, s in enumerate(gene_symbol)}

    sym_rows = defaultdict(list)
    for row_i, s in enumerate(symbols):
        sym_rows[s].append(row_i)

    G = len(gene_symbol)
    score_cells_matrix = np.zeros((G, n_ct), dtype=np.float64)
    for s, rows in sym_rows.items():
        score_cells_matrix[sym_to_idx[s], :] = genes_cells_score[rows, :].max(axis=0)

    score_cells_matrix = scipy.sparse.csr_matrix(score_cells_matrix)

    RP_adata = ad.AnnData(score_cells_matrix.T,
                          obs=pd.DataFrame(index=cells_list),
                          var=pd.DataFrame(index=gene_symbol))
    return RP_adata



def monitor_runtime_memory(func):
    """
    Decorator to monitor runtime and peak memory usage of a function.
    Always logs runtime and memory stats using the function's internal logger.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2  # in MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024**2
        end_time = time.time()
        runtime = end_time - start_time
        mem_used = mem_after - mem_before

        msg = (
            f"\n⏱ Runtime: {runtime:.2f} seconds\n"
            f"💾 Memory usage increase: {mem_used:.2f} MB (RSS)\n"
            f"   Memory before: {mem_before:.2f} MB, after: {mem_after:.2f} MB\n"
        )

        # Always assume logger is created inside the function
        logger.info(msg)

        return result

    return wrapper



def process_tf(tf, feature_matrix, chip_index, ref_baseline, remove_others, decay,
               model, refgene_path, n_cores, n_factors, corr_csv_template=None):
    logger.info(f"### Starting {tf} ###")
    chip_peak_adata = impute(
        input_mat_adata=feature_matrix,
        impute_factor=tf,
        ref_path=chip_index,
        ref_baseline=ref_baseline,
        remove_others_source=remove_others,
        n_cores=max(1, n_cores // max(1, n_factors))  # divide cores per TF
    )
    logger.info(f"Calculating gene-level RP matrix for {tf}...")
    gene_matrix = count_to_gene_by_RP(
        input_adata=chip_peak_adata,
        decay=decay,
        model=model,
        refgene_path=refgene_path,
        corr_csv_template=corr_csv_template,   # per-cell-type pruning template
    )
    logger.info(f"Finished {tf}")
    return tf, gene_matrix
