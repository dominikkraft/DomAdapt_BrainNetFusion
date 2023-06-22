"""
FUNCTIONS for similarity network fusion, diffusion map embedding and domain adapted ML
Author: Dominik Kraft
"""

from brainspace.gradient import embedding
from adapt.instance_based import TrAdaBoostR2
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mapalign import align
from snf import compute
import numpy as np


def fusion(modalities, K=30, mu=0.8):
    """
    This functions 1) calculates affinity matrices for each single modality and 2)
    performs similarity network fusion on affinity matrices.

    Input:
    - modalities: list of numpy arrays (n_subjects, n_features) per modality, e.g., area, volume
    - K: number of neighboors used in calculating affinity matrices and in fusion process
    - mu: scaling factor, ### tbaa ###

    Output:
    - fused similarity network matrix; shape n_subjects x n_subjects
    - list of affinity matrices
    """

    to_fuse = []

    for modality in modalities:
        affinity = compute.make_affinity(modality, K=K, mu=mu)
        assert affinity.shape == (
            modality.shape[0],
            modality.shape[0],
        ), "affinity dimensions wrong"
        to_fuse.append(affinity)
    fused = compute.snf(to_fuse, K=30)

    return fused, to_fuse


def diff_map_embedding(fused, n_components=10):
    """
    This function performs diffusion map embedding
    ------
    Input:
    - fusion = similarity fused network (nsubjects x nsubjects)
    - n_components = 10 (Default), number of components to estimate

    ------
    Output:
    - eigenvectors: array shape subjects x n_components
    - eigenvalues: array shape subjects x n_components
    """

    evcs, evals = embedding.diffusion_mapping(fused, n_components=n_components)

    return evcs, evals


# functions for prediction framework


def prepare_mean_features(array_list):
    """
    This function calculates the mean across the left and right hemisphere per feature MRI modality
    ------
    Input:
    - array_list: list of len(dataframes) containing numpy arrays (nsubjects x 68)
    ------
    Output:
    - features: np.array (nsubjects x 68) build from mean MRI modalities across hemispheres
    """

    temp_list = []
    # get number of features per hemisphere
    hemi_features = int(array_list[0].shape[1] / 2)

    for i in range(len(array_list)):
        feat = (array_list[i][:, :hemi_features] + array_list[i][:, hemi_features:]) / 2
        temp_list.append(feat)

    features = np.hstack(tuple(temp_list))

    return features, temp_list


def supervised_domain_adapt(
    train_features, train_y, target_features, target_y, test_features, model
):
    """
    This function uses supervised Domain Adaptation to enhance the prediction of
    the low dimensional manifold in unseen data.
    ------
    Input:
    - train_features: MRI features from PNC, np.array (nsubjects x 68) (source)
    - train_y: first PNC embedding, np.array (nsubjects,) (source)
    - target_features: feature data from target used for DA
    - target_y: y data from target used for DA
    - test_features: unseen data from target used for prediction
    - model= default: ElasticNet()), sklearn model
    ------
    Output:
    - predictions: predicted embedding - np.array (nsubjectsx1)
    """

    DA = TrAdaBoostR2(
        model,
        n_estimators=10,
        Xt=target_features,
        yt=target_y,
        random_state=123,
        verbose=0,
    )
    DA.fit(train_features, train_y)
    predictions = DA.predict(test_features)

    return predictions


def procustes_alignment(predictions, true_emb):
    """
    Perform procrustes alignment with respect to 'ground truth'
    ------
    Input:
    - aligned: predicted embedding after procrustes alignment
    - true embedding
    ------
    Output:
    - aligned_emb = predicted embedding aligned to 'ground truth'
    """

    aligned_pred = align.iterative_alignment(
        [true_emb.reshape(-1, 1), predictions.reshape(-1, 1)]
    )[0][1]
    return aligned_pred


def calc_metrics(aligned_predictions, true_emb, print_metrics=True):
    """
    Calculates metrics for prediction model
    ------
    Input:
    - aligned: predicted embedding after procrustes alignment
    - true embedding
    - print_metrics: True (default), whether metrics should be printed
    ------
    Output:
    - metrics: List containing result scores in 5 metrics
    """

    r, _ = pearsonr(true_emb, aligned_predictions)
    mse = mean_squared_error(true_emb, aligned_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_emb, aligned_predictions)
    r2 = r2_score(true_emb, aligned_predictions)

    if print_metrics == True:
        print(
            f" r={round(r, 3)}, mse={round(mse, 3)}, rmse={round(rmse, 3)}\
 ,mae={round(mae, 3)}, r2={round(r2,3)}"
        )

    return [r, mse, rmse, mae, r2]
