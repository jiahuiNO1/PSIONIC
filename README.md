# PSIONIC

![tests](https://github.com/jiahuiNO1/PSIONIC/actions/workflows/tests.yml/badge.svg)

**PSIONIC** infers cell-type-specific transcription factor (TF) activity by integrating
ChIP-seq-supported regulatory priors with single-cell chromatin accessibility (scATAC-seq)
and gene expression (scRNA-seq).

This package is the Python implementation of PSIONIC
([Osmanbeyoglu et al., *Nature Communications*, 2019](https://doi.org/10.1038/s41467-019-12291-6)),
whose original MATLAB code is at [osmanbeyoglulab/PSIONIC](https://github.com/osmanbeyoglulab/PSIONIC),
extended to infer TF activity at cell-type resolution.

## How it works

PSIONIC has two main steps:

1. **Build cell-type-specific TF–gene priors (`D`) from scATAC-seq.**
   For each cell type, accessible peaks from scATAC-seq pseudobulk are matched to experimental
   TF binding signals from ChIP-seq references and linked to target genes with regulatory
   potential scores. The result is a TF × gene matrix for each cell type.

2. **Infer TF activity with a multitask model on scRNA-seq.**
   Each cell type–sample expression profile is one prediction task. PSIONIC models every task
   as a combination of shared latent regulatory programs (`L`, TFs × programs) with
   task-specific weights (`S`, programs × tasks), learned jointly with regularised multitask
   optimisation (GO-MTL). TF activity for each task is given by `L × S`.

## Installation

PSIONIC requires **Python 3.11 or later**.

### From source (recommended)

Installing from source keeps the `conf/` reference folder next to the code:

```bash
git clone https://github.com/jiahuiNO1/PSIONIC.git
cd PSIONIC
pip install -e .
```

### Conda environment

To use the exact package versions used in development:

```bash
conda env create -f environment.yml
conda activate psionic
```

## Reference data

`create_TF_target_matrix` reads its reference files from the `conf/` folder inside the
package, `src/psionic/conf/`. The configuration file and gene annotations are included
in the repository; the ChIP-seq reference BED files are downloaded separately (below).
When set up, the folder looks like this:

```
src/psionic/conf/
├── config.yml
├── GRCh38_refgenes.txt      # human gene annotation (included)
├── GRCm38_refgenes.txt      # mouse gene annotation (included)
├── TR_human/                # downloaded, see below
│   ├── raw_beds/            # ChIP-seq reference BED files used by PSIONIC
│   └── cache.*.dat / .idx   # SCRIP's GIGGLE search index (not used by PSIONIC), can remove
└── TR_mouse/                # downloaded, see below
    ├── raw_beds/
    └── cache.*.dat / .idx
```

### Download the ChIP-seq references

The ChIP-seq reference BED files are the transcription regulator (TR) references from
SCRIP, available on Zenodo ([doi:10.5281/zenodo.5840810](https://doi.org/10.5281/zenodo.5840810),
CC BY 4.0). Download and extract the archive for your species inside `src/psionic/conf/`;
each archive extracts to a `TR_human/` or `TR_mouse/` folder:

```bash
cd src/psionic/conf

# Human (2.2 GB)
wget https://zenodo.org/records/5840810/files/TR_human.tar.gz
md5sum TR_human.tar.gz      # expected: 97faa2aa8db96abd8b529ed214083707
tar -xzf TR_human.tar.gz && rm TR_human.tar.gz

# Mouse (1.5 GB)
wget https://zenodo.org/records/5840810/files/TR_mouse.tar.gz
md5sum TR_mouse.tar.gz      # expected: c7d2650bfff49b123dd24356836eb6c4
tar -xzf TR_mouse.tar.gz && rm TR_mouse.tar.gz
```

## Quick start

```python
import psionic

psionic.set_log_file("tf_imputation_full.log")   # optional: write a log file

if __name__ == "__main__":   # required in scripts: TFs are processed in parallel

    # Step 1. TF x gene prior matrix (D) for each cell type
    results = psionic.create_TF_target_matrix(
        input_matrix_path="pseudobulk_count_matrix/",   # cell type x peak counts
        factors=["CTCF", "SPI1", "REST"],
        species="human",
        n_cores=8,
    )
    D = psionic.combine_tf_results(results, out_dir="TFxGene/")   # {cell type: AnnData}

    # Step 2. Multitask model: one task per cell type-sample expression profile
    #   X_train[t]: gene x TF prior matrix for the cell type of task t
    #   Y_train[t]: gene x 1 expression vector, column-normalised with psionic.normalize_column
    L, S, W = psionic.GO_MTL(X_train, Y_train, k=5, Maxsteps=20,
                             lambda_=0.005, mu=0.5, O="R")
```

## Tutorials

| Notebook | Contents |
|---|---|
| [`PSIONIC_tutorial_part1.ipynb`](tutorials/PSIONIC_tutorial_part1.ipynb) | Build `D` from scATAC-seq pseudobulk, build `Y` from scRNA-seq, tune hyperparameters with 5-fold cross-validation |
| [`PSIONIC_tutorial_part2.ipynb`](tutorials/PSIONIC_tutorial_part2.ipynb) | Fit the full model, interpret `S` and `L`, run differential TF activity analysis |

The toy inputs used by the tutorials are in `tutorials/data/`:

- `toy_pseudobulk_count_matrix/`
- `selected_50_TFs.csv`
- `toy_Y.h5ad`
- `metadata.csv`

To run the notebooks, install the extra tutorial dependencies (Jupyter and session-info):

```bash
pip install -e ".[tutorials]"
```

## Logging

PSIONIC writes nothing by default. To record progress messages in a file:

```python
psionic.set_log_file("my_run.log")             # overwrite (default)
psionic.set_log_file("my_run.log", mode="a")   # append
```

When running several jobs from the same folder, give each job its own log file.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use PSIONIC, please cite the original PSIONIC paper:

> Osmanbeyoglu HU, Shimizu F, Rynne-Vidal A, Alonso-Curbelo D, Chen H-A, Wen HY, Yeung T-L,
> Jelinic P, Razavi P, Lowe SW, Mok SC, Chiosis G, Levine DA, Leslie CS.
> Chromatin-informed inference of transcriptional programs in gynecologic and basal breast cancers.
> *Nature Communications* 10, 4369 (2019). https://doi.org/10.1038/s41467-019-12291-6
<!-- TODO: add the reference for the single-cell PSIONIC paper when published -->

GitHub's **Cite this repository** button (right sidebar) uses [`CITATION.cff`](CITATION.cff).

## Acknowledgements

- The multitask model follows GO-MTL: Kumar A. & Daumé III H. *Learning Task Grouping and
  Overlap in Multi-task Learning.* ICML 2012.
- The optimisation routines in `gomtl_helpers.py` (`minFunc`, `WolfeLineSearch`,
  `L1General2_PSSgb`, ...) are Python ports of Mark Schmidt's MATLAB packages
  minFunc and L1General.
- The construction of TF–gene priors (reference matching, imputation and regulatory
  potential) is adapted from SCRIP: Dong X, Tang K, Xu Y, Wei H, Han T & Wang C.
  *Single-cell gene regulation network inference by large-scale data integration.*
  Nucleic Acids Research 50, e126 (2022). SCRIP's code is released under the MIT License
  (see [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md)). The ChIP-seq reference BED files
  are SCRIP's reference datasets ([doi:10.5281/zenodo.5840810](https://doi.org/10.5281/zenodo.5840810)).

## License

PSIONIC is released under the MIT License. See [`LICENSE`](LICENSE). Licence notices for third-party code included in PSIONIC are in
[`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md).

## Authors

Jiahui Hou, Parham Hadikhani and Hatice U. Osmanbeyoglu, [Osmanbeyoglu Lab](https://www.osmanbeyoglulab.com),
University of Pittsburgh.

## Contact

Jiahui Hou ([jih325@pitt.edu](mailto:jih325@pitt.edu)).
Please report bugs and questions through [GitHub Issues](https://github.com/jiahuiNO1/PSIONIC/issues).
