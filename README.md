# DomAdapt_BrainNetFusion

This repository stores code and analyses behind **preprint** 

Before running the code, please make sure to set up a new conda environment first:

`conda create --name <env> --file requirements.txt`

Make sure to activate env for all analyses within this repo: `conda activate <env>`


## Order of Operations

- `main_abcd.py` and `main_hbn.py` are the backbone of the analyes, performing all steps from data loading to machine learning with domain adaptation
- the remaining `.py` files store functions used in `main_abcd.py` and `main_hbn.py`
- association analyses with pubertal and mental health measures were performed in the `*.ipynb` files. 
