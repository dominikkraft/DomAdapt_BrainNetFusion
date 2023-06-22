"""
Main Script containing the following steps:
- perform similarity network fusion and diffusion map embedding in PNC
- train ML model on PNC embeddings with supervised Domain Adaptation
- HBN *no diagnosis* data enhances prediction of embedding in unseen HBN data
Author: Dominik Kraft
"""

import time
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from utils_hbn import load_loris_data, load_coins_data
from utils import create_data_array, site_harmonization, load_mri
from analyze import (
    fusion,
    diff_map_embedding,
    prepare_mean_features,
    supervised_domain_adapt,
    procustes_alignment,
    calc_metrics,
)


exclude_ROIs = [
    "CSF",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-VentralDC",
    "Right-vessel",
    "Right-choroid-plexus",
    "5th-Ventricle",
    "WM-hypointensities",
    "Left-WM-hypointensities",
    "Right-WM-hypointensities",
    "non-WM-hypointensities",
    "Left-non-WM-hypointensities",
    "Right-non-WM-hypointensities",
    "Optic-Chiasm",
    "BrainSegVol",
    "BrainSegVolNotVent",
    "lhCortexVol",
    "rhCortexVol",
    "CortexVol",
    "lhCerebralWhiteMatterVol",
    "rhCerebralWhiteMatterVol",
    "CerebralWhiteMatterVol",
    "SubCortGrayVol",
    "TotalGrayVol",
    "SupraTentorialVol",
    "SupraTentorialVolNotVent",
    "MaskVol",
    "BrainSegVol-to-eTIV",
    "MaskVol-to-eTIV",
    "lhSurfaceHoles",
    "rhSurfaceHoles",
    "SurfaceHoles",
    "BrainSegVolNotVent",
    "eTIV",
    "rh_MeanThickness_thickness",
    "lh_MeanThickness_thickness",
    "EstimatedTotalIntraCranialVol",
    "lh_WhiteSurfArea_area",
    "rh_WhiteSurfArea_area",
]


print("Starting script ....")

start_time = time.time()

###########
## PNC ####
##########

# load, concat data -> fusion and embedding
pnc_area, pnc_volume = load_mri(dataset="PNC", exclude=exclude_ROIs)
pnc_arrays = create_data_array([pnc_area, pnc_volume])
fused_pnc, _ = fusion(pnc_arrays)
pnc_evc, pnc_eval = diff_map_embedding(fused_pnc)

# prepate features for ML framework
features_pnc, _ = prepare_mean_features(pnc_arrays)


print("PNC: Done!")
############
## HBN ####
############

print("HBN: Starting!")

hbn_area, hbn_volume = load_mri(dataset="HBN", exclude=exclude_ROIs)

# load non-MRI data from HBN that includes demographics, mri site & diagnoses
# we use data-uploads from LORIS and COINS and merge them into a single file

loris = load_loris_data()
coins = load_coins_data()

demos = pd.concat([loris, coins])
demos.drop_duplicates(subset=["Identifiers"], inplace=True)

# calculate the summed diagnoses as proxy for psychopathology severity
# we substract 4 because of the 4 columns not related to diagnostics

demos["summed"] = (
    demos.loc[
        ~demos["DX_01"].isin(
            ["No Diagnosis Given", "No Diagnosis Given: Incomplete Eval"]
        )
    ].count(axis=1)
    - 4
)


## adjust sample
# use only subjects that are in MRI data
demos = (
    demos.loc[lambda row: row["Identifiers"].isin(hbn_area["subjectkey"].to_list())]
    .sort_values("Identifiers")
    .reset_index(drop=True)
)
# which subjects from MRI data are not in demos and thus need to be excluded from MRI?
excl_fromMRI = [
    x
    for x in hbn_area["subjectkey"].to_list()
    if x not in demos["Identifiers"].to_list()
]

hbn_area = hbn_area.loc[lambda row: ~row["subjectkey"].isin(excl_fromMRI)].reset_index(
    drop=True
)
hbn_volume = hbn_volume.loc[
    lambda row: ~row["subjectkey"].isin(excl_fromMRI)
].reset_index(drop=True)

assert (
    demos["Identifiers"].to_list()
    == hbn_area["subjectkey"].to_list()
    == hbn_volume["subjectkey"].to_list()
), "diff_subjects"


# initialize empty list for output
embeddings_hbn = []
features_hbn = []
demograph = []
predictions = []
metrics = []

# subsetting with "healthy subjects" used for domain adaptation
subset = np.array(demos["summed"].isnull()).nonzero()[0]


demos_sub = demos.iloc[demos.index.isin(subset)]
hbn_area_sub = hbn_area.iloc[hbn_area.index.isin(subset)]
hbn_vol_sub = hbn_volume.iloc[hbn_volume.index.isin(subset)]

assert (
    demos_sub["Identifiers"].to_list() == hbn_area_sub["subjectkey"].to_list()
), "diff subs"

# test subjects
d = demos.iloc[~demos.index.isin(subset)]
hbn_a = hbn_area.iloc[~hbn_area.index.isin(subset)]
hbn_v = hbn_volume.iloc[~hbn_volume.index.isin(subset)]

assert d["Identifiers"].to_list() == hbn_a["subjectkey"].to_list(), "diff subs"


# create list with healthy and patients list containing demo, area, volume
hbn_list = [[demos_sub, hbn_area_sub, hbn_vol_sub], [d, hbn_a, hbn_v]]


for i in range(len(hbn_list)):
    site_info = hbn_list[i][0]

    print(site_info.shape)  # prints number of subjects

    data_input = hbn_list[i][1:]  # ignore first entry == demo
    assert len(data_input) == 2, "not enough dataframes!"

    # transform DF into np.arrays and harmonize MRI data
    array_list = create_data_array(data_input)
    array_list_combat = site_harmonization(array_list, site_info, "Scan_Location")

    # fusion, embedding
    print("HBN Fusion: Starting!")
    fused_hbn, _ = fusion(array_list_combat)
    hbn_evc, _ = diff_map_embedding(fused_hbn)
    embeddings_hbn.append(hbn_evc)

    hbn_feats, _ = prepare_mean_features(array_list_combat)
    features_hbn.append(hbn_feats)

    demograph.append(site_info)


print("Starting Prediction Framework ... ")

preds = supervised_domain_adapt(
    train_features=features_pnc,
    train_y=pnc_evc[:, 0],
    target_features=features_hbn[0],
    target_y=embeddings_hbn[0][:, 0],
    test_features=features_hbn[1],
    model=ElasticNet(),
)

aligned_prediction = procustes_alignment(preds, embeddings_hbn[1][:, 0])

mets = calc_metrics(np.squeeze(aligned_prediction), embeddings_hbn[1][:, 0])

metrics.append(mets)
predictions.append(preds)

print("saving file workind dir")
# prepate output dict
output = {
    "demos": demograph,
    "embeddings": embeddings_hbn[1:],
    "predictions": predictions,
    "metrics": metrics,
}
np.save("outputdict_hbn.npy", output)

print("Finishing Pipeline ... ")
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
