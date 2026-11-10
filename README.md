# AIDD8122 Final Project

## Reproduction and Evaluation of an HG-Only XPert Variant for Drug-Induced Transcriptomic Response Prediction

**Student:** Qiu Dian  
**Student ID:** P2600216  
**Institution:** Macao Polytechnic University  
**Course:** AIDD8122 – Advanced Topics in Artificial Intelligence & Drug Discovery

### 1. Project Overview

This course project evaluates a modified HG-only variant of XPert for predicting drug-induced transcriptomic responses.

The central research question is whether XPert can predict meaningful drug-induced gene expression changes beyond a simple no-change baseline.

### 2. Dataset

The study uses a subset of the publicly available LINCS L1000 perturbation dataset provided by the XPert authors.

| Dataset split | Number of samples |
|---|---:|
| Training | 1,000 |
| Validation | 200 |
| Testing | 200 |
| Total | 1,400 |

Each sample contains expression measurements for 978 genes.

**Data source:** https://zenodo.org/records/17182939

### 3. Methods

The experiment uses an adapted XPert model with knowledge-graph-based drug representations.

The UniMol branch was disabled, and FlashAttention was replaced with a PyTorch-compatible attention implementation for NVIDIA Tesla T4.

Training used a simplified MSE-based objective for 10 epochs. Model selection was based on validation-set MSE.

A no-change baseline was used for comparison.

### 4. Main Results

| Metric | No-change Baseline | XPert-HG-only |
|---|---:|---:|
| MSE | 0.519923 | 0.510133 |
| RMSE | 0.721057 | 0.714236 |
| MAE | 0.442325 | 0.445866 |

The adapted XPert model reduced test MSE by approximately 1.88%.

The mean Pearson correlation for predicted versus observed expression changes was 0.132.

MSE improved for 136 of the 200 test samples.

Drug-cluster bootstrap analysis estimated a mean MSE reduction of 0.009791, with a 95% confidence interval of [0.007556, 0.012155].

### 5. Project Files

- `notebooks/` — Colab notebook containing the experimental workflow.
- `models/` — Adapted XPert model source code.
- `configs/` — Experimental configuration files.
- `train_xpert.py` and `utils.py` — Adapted training and data-loading code.
- `results/` — Training history, model comparison, bootstrap statistics, and saved predictions.
- `figures/` — Experimental visualizations.

### 6. Limitations

This is an adapted HG-only implementation, not a full reproduction of the original XPert architecture and training protocol.

The experiment uses a 1,400-sample subset, a single training run, and a simplified loss function. The model shows limited correlation with observed drug-induced expression changes.

The unseen-drug subgroup analysis is exploratory and does not constitute a strict cold-drug benchmark.

### How to Reproduce This Project

**Recommended environment:** Google Colab with an NVIDIA GPU.

**Main reproduction notebook:**

`notebooks/AIDD8122_XPert_Reproduction_CLEAN.ipynb`

**Steps:**

1. Open the cleaned notebook in Google Colab.
2. Enable a GPU runtime.
3. Follow the notebook to set up dependencies and obtain the official XPert dataset.
4. Generate the 1,400-sample experimental subset using the fixed random seed (2026).
5. Prepare the HG-only model configuration and drug feature interface.
6. Train the adapted XPert model for 10 epochs, selecting the best checkpoint by validation MSE.
7. Evaluate the checkpoint on the held-out test set against the no-change baseline.
8. Generate the model comparison tables, figures, and drug-cluster bootstrap confidence interval.

**Experimental outputs:**

- `results/training_history_large.csv`
- `results/model_comparison_large.csv`
- `results/bootstrap_results.csv`
- `results/test_predictions_large.npz`
- `results/figures/` — Experimental figures, if stored at this path.

**Reproducibility notes:**

This project uses an adapted HG-only XPert implementation rather than the full original model. The UniMol branch is disabled, and a simplified training objective is used.

The cleaned notebook has been organized and syntax-checked, but the complete training pipeline has not been independently rerun in a fresh environment. The reported numerical results originate from the completed experimental run.


## Pearson-Loss Ablation Study

An exploratory ablation study was conducted to evaluate
the contribution of Pearson correlation-based supervision
to drug-induced transcriptomic response prediction.

On 200 held-out test observations, adding the Pearson loss
reduced MSE from 0.510133 to 0.506850 and improved mean
Pearson delta from 0.131860 to 0.154565.

The improvement is exploratory because the two conditions
were trained in different hosted environments, with one
training run per condition.

- [Ablation methods and findings](README_ABLATION.md)
- [Kaggle reproduction notebook](notebooks/AIDD8122_XPert_Pearson_Ablation_Kaggle.ipynb)
- [Ablation comparison results](results/ablation_comparison.csv)


### 7. Original Work and Attribution

This repository is forked from the original XPert project.

**Original XPert repository:** https://github.com/GSanShui/XPert

The original authors' contributions, documentation, and license are retained below.

---

## Original XPert Documentation

*The original repository README continues below this section.*
# XPert: Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch Transformer

## Introduction
Systematic mapping of chemical perturbation responses is revolutionizing polypharmacological drug discovery, yet remains constrained by experimental scalability. Here, we introduce XPert, a biologically informed Dual-Branch Transformer model that predicts gene-specific drug responses across dose-time conditions, outperforming VAE-based methods. It generalizes to unseen cells, transfers knowledge to clinical settings, and reveals mechanistic insights, offering a scalable solution for precision medicine and perturbation-based drug discovery. 

## Overview
The repository is organised as follows:
- `processed_data/` contains processed data files;
- `HG_data/` contains the data for training the knowledge heterogeneous graph and the pre-trained embeddings;
- `dataset/` contains the necessary files for creating the dataset;
- `models/` contains different modules of XPert;
- `configs/` contains all the config files adapted for different datasets and scenarios;
- `experiment/` contains log files and output files;
- `scripts/` contains the scripts for training, inference, and testing the model;
- `saved_model/` contains the trained and pretrained weights;
- `evaluation_metrics/` contains all the evaluation metrics mentioned in the paper;
- `reproducing/` contains the code for reproducing the analysis results and figures from the paper;

## Requirements

### OS Requirements
The package development version is tested on Linux: Ubuntu 20.04 operating systems.

### Python Dependencies
The XPert network is built using PyTorch. You can use following commands to create conda env with related dependencies.
``` 
conda create -n xpert python=3.9
pip install -r requirements.txt
``` 

## Example
If you just need to quickly test the code with a demo, we provide a small example dataset that you can run using the following command. For testing with the full dataset and reproducing the results presented in the paper, please refer to the next section on [Implementation](#implementation).
``` 
bash scripts/train_example.sh
``` 

## Implementation

**Note:** Due to file size limitations, some files are not included in this repository. Please refer to [Zenodo](https://doi.org/10.5281/zenodo.15357711
        
        
        
        ) and [Figshare](https://doi.org/10.6084/m9.figshare.28955141
        
        
        
        ) for the complete set of resources.

### Data Format
1. Paired pre- and post-treatment datasets are stored in `h5ad` format. The post-treatment gene expression data is stored in `adata.X`, while the pre-treatment data is stored in `adata.obsm["X_ctl"]`, and metadata is stored in `adata.obs`. All datasets used for XPert follow this format.

2. The processed datasets can be accessed at Figshare (URL to be updated after paper acceptance).

### Model Training
The scripts for training, inference, and testing the model are located in the `scripts/` folder. Example usage is shown below:

Run the following command to train XPert using the L1000 dataset (for predictions in multi-dose-multi-time mode, change the dataset suffix to `_mdmt`, e.g., `l1000_mdmt`). Set the `use_gradscaler` parameter to `True` to deploy flash attention for accelerating the attention computation.

``` 
python train_xpert.py --model XPert 
                      --config config_l1000
                      --drug_feat unimol
                      --nfold split_1
                      --dataset l1000_sdst
                      --use_gradscaler True
``` 


### Model Finetuning

Our paper demonstrates the powerful potential of using the L1000 dataset as a pre-training dataset. If you wish to extend this application to other datasets, we recommend using the finetuning mode.

All trained models are available for download on [Figshare](URL to be updated after paper acceptance). Please store them in the `saved_models/` folder. Modify the config file for different datasets and specify the path to the pre-trained model using the `pretrained_model` parameter.


For cold-dose-time scenarios, use the following code:

```
python train_xpert.py --model XPert 
                      --config config_l1000_cdt 
                      --drug_feat unimol
                      --nfold split_cold_dose&time_random_1_one_shot
                      --dataset l1000_mdmt
                      --use_gradscaler True
                      --pretrained_model saved_model/pretrain_mdmt_new.pth
```

For independent datasets, use the following code:
```
python train_xpert.py --model XPert 
                      --config config_cdsdb
                      --drug_feat unimol
                      --nfold split_breast_1
                      --dataset cdsdb_mdmt
                      --use_gradscaler True
                      --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth
```

### Model Testing/Inferencing

You can use our saved model for result reproduction or novel prediction. Please set the `output_profile` parameter to `True` if you wish to output the prediction profile.


``` 
python train_xpert.py --model XPert 
                      --config config_l1000
                      --drug_feat unimol
                      --nfold split_1
                      --dataset l1000_sdst
                      --use_gradscaler True
                      --mode test
                      --output_profile True
                      --saved_model_path saved_model/l1000_sdst_warm_split.pth
``` 

If you need to extract `cls` embeddings or attention matrices for downstream analysis, we recommend running the model in `Infer` mode. Set the `output_cls_embed` and `output_attention` parameters accordingly. For step-by-step guidance on attention analysis, refer to the tutorials at `reproducing/fig3/fig3D_egfr_attn_molecule.ipynb` and `reproducing/fig6/Fig6G_differential_atttention_analysis.ipynb`. These notebooks provide reproducible examples of how to interpret attention outputs.

``` 
python train_xpert.py --model XPert 
                      --config config_l1000
                      --drug_feat unimol
                      --nfold split_1
                      --dataset l1000_sdst
                      --use_gradscaler True
                      --mode infer
                      --output_cls_embed True
                      --output_attention True
                      --saved_model_path saved_model/l1000_sdst_warm_split.pth
``` 



