# DomAdapt_BrainNetFusion

This repository stores code and analyses for the [preprint](https://www.medrxiv.org/content/10.1101/2023.01.26.23285055v1) entitled 

> *Signatures of pubertal brain development and health revealed through domain adapted brain network fusion* 

Before running the code, please make sure to set up a new conda environment first:

- `conda create --name <env> python=3.7.11`
- `conda activate <env>`
- `pip install -r requirements.txt`

Make sure to activate env for all analyses within this repo: `conda activate <env>`


## Order of Operations

- `main_abcd.py` and `main_hbn.py` are the backbone of the analyes, performing all steps from data loading to machine learning with domain adaptation
- the remaining `.py` files store functions used in `main_abcd.py` and `main_hbn.py`
- association analyses with pubertal and mental health measures were performed in the `ABCD_*.ipynb and HBN_*.ipynb` files.
- `Xemb.ipynb` contains code that calculates feature - embedding correlations for brain map vizualisations and spin permutation testing
- `*.R` uses output from `Xemb.ipynb` and generates brain maps for Suppl. Materials (ggseg 1.6.5 running in R 4.2.3)



[![DOI](https://zenodo.org/badge/591243629.svg)](https://zenodo.org/badge/latestdoi/591243629)

