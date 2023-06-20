"""
Main Script containing the following steps:
- perform similarity network fusion and diffusion map embedding in PNC
- split ABCD data into baseline (only T1), baseline, follow up
- train ML model on PNC embeddings with supervised Domain Adaptation
- ABCD *baseline only* data enhances prediction of embedding in unseen ABCD data
Author: Dominik Kraft
"""

import time
import numpy as np
from sklearn.linear_model import ElasticNet
from utils import (
    create_data_array,
    site_harmonization,
    load_abcd_mri,
    extend_demodf,
    load_mri,
    euler_exclude,
)
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


###################################################################
# data quality follow up analysis #################################
# excluding PNC subjects based on euler number (3xSD below mean) #
# (un)comment the next 4 lines ####################################
################################################################### 

#euler_mask = euler_exclude(pnc_area)
#print(euler_mask.sum())
#pnc_area = pnc_area[~euler_mask]
#pnc_volume = pnc_volume[~euler_mask]


pnc_arrays = create_data_array([pnc_area, pnc_volume])
fused_pnc, _ = fusion(pnc_arrays)
pnc_evc, pnc_eval = diff_map_embedding(fused_pnc)

# prepate features for ML framework
features_pnc, _ = prepare_mean_features(pnc_arrays)


print("PNC: Done!")
############
## ABCD ####
############

print("ABCD: Starting!")

abcd_list = load_abcd_mri()

# extend demographic dfs with additional variables, here: scanner site
abcd_list = extend_demodf(abcd_list, "abcd_mri01", ["mri_info_deviceserialnumber"])


# initialize empty list for output
embeddings_abcd = []
features_abcd = []
demos = []
predictions = []
metrics = []

for i in range(len(abcd_list)):
    site_info = abcd_list[i][0]

    print(site_info.shape)

    data_input = abcd_list[i][1:]  # ignore first df entry == demo
    assert len(data_input) == 2, "not enough dataframes!"

    # transform DF into np.arrays
    array_list = create_data_array(data_input)
    array_list_combat = site_harmonization(
        array_list, site_info, "mri_info_deviceserialnumber"
    )

    # fusion, embedding
    print("ABCD Fusion: Starting!")
    fused_abcd, _ = fusion(array_list_combat)
    abcd_evc, _ = diff_map_embedding(fused_abcd)
    embeddings_abcd.append(abcd_evc)

    abcd_feats, _ = prepare_mean_features(array_list_combat)
    features_abcd.append(abcd_feats)

    demos.append(site_info)

#####################
#### ML Framework ###
#####################


print("Starting Prediction Framework ... ")

for feat in range(1, 3):

    if feat == 1:
        print(".. for baseline")
    elif feat == 2:
        print(".. for 2 years follow up")

    preds = supervised_domain_adapt(
        train_features=features_pnc,
        train_y=pnc_evc[:, 0],
        target_features=features_abcd[0],
        target_y=embeddings_abcd[0][:, 0],
        test_features=features_abcd[feat],
        model=ElasticNet(),
    )

    aligned_prediction = procustes_alignment(preds, embeddings_abcd[feat][:, 0])

    mets = calc_metrics(np.squeeze(aligned_prediction), embeddings_abcd[feat][:, 0])

    metrics.append(mets)
    predictions.append(preds)

print("saving file in working dir")
# prepate output dict
output = {
    "demos": demos[1:],
    "embeddings": embeddings_abcd[1:],
    "predictions": predictions,
    "metrics": metrics,
}
np.save("outputdict_base_followup_eulerexcl.npy", output)

print("Finishing Pipeline ... ")
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
