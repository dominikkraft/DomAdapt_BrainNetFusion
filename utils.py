"""
General utility functions and ABCD specific util functions
Author: Dominik Kraft
"""

import pandas as pd
import numpy as np
from neuroCombat import neuroCombat


def load_mri(dataset, exclude):
    """
    This function loads MRI data, either from PNC or HBN
    ------
    Input:
    - dataset = either PNC or HBN (string)
    - exclude = list of ROIs that should be excluded, e.g. CSF
    ------
    Output:
    - area: df (subjects x 69) - 68 ROI area + subjects identifier
    - volume: df (subjects x 69) - 68 ROI volume + subjects identifier

    """

    if dataset == "PNC":
        path = "../PNC/MRI/"
    elif dataset == "HBN":
        path = "../hbn/MRI/new0911/"

    # load area and volume dataframes per LH / RH
    lh_area = pd.read_csv(
        path + "lh.area.txt", sep="\t", usecols=lambda x: x not in exclude
    )
    lh_volume = pd.read_csv(
        path + "lh.volume.txt", sep="\t", usecols=lambda x: x not in exclude
    )
    rh_area = pd.read_csv(
        path + "rh.area.txt", sep="\t", usecols=lambda x: x not in exclude
    )
    rh_volume = pd.read_csv(
        path + "rh.volume.txt", sep="\t", usecols=lambda x: x not in exclude
    )

    # perform minimal dataframe wrangling
    for df in [lh_area, lh_volume, rh_area, rh_volume]:
        df.rename({df.columns[0]: "subjectkey"}, axis=1, inplace=True)
        df.drop(labels=[col for col in df.columns if "??" in col], axis=1, inplace=True)

    # merge LH and RH dataframes per modality
    area = (
        pd.merge(lh_area, rh_area, on="subjectkey")
        .sort_values("subjectkey")
        .reset_index(drop=True)
    )
    volume = (
        pd.merge(lh_volume, rh_volume, on="subjectkey")
        .sort_values("subjectkey")
        .reset_index(drop=True)
    )

    assert (
        volume["subjectkey"].to_list() == area["subjectkey"].to_list()
    ), "different subjects!!"
    return area, volume


def load_abcd_mri():
    """
    This function loads MRI data from the ABCD study and
    ------
    Input:
    -
    ------
    Output:
    - output_list: list containing demo, area, vol dfs for baseline_only and two timepoint (BL, T1) subjects
    """
    # load header info and store in dictionary
    header = pd.read_csv(
        "../network_fusion/Package_1197473/abcd_smrip10201.txt",
        header=None,
        sep="\t",
        nrows=2,
    )  # read header
    # load data and use header info to name columns
    data = pd.read_csv(
        "../network_fusion/Package_1197473/abcd_smrip10201.txt",
        header=None,
        sep="\t",
        skiprows=2,
    )  # read data
    data.columns = list(header.iloc[0, :])

    # drop duplicates
    data = data.drop_duplicates(
        subset=["subjectkey", "interview_date", "interview_age"], ignore_index=False
    )

    # select columns of interest
    democols = ["subjectkey", "interview_age", "sex", "eventname"]
    areacols = [
        col for col in data.columns if ("area_cdk" in col) & ("total" not in col)
    ]
    volcols = [col for col in data.columns if ("vol_cdk" in col) & ("total" not in col)]

    # subset data with columns of interest
    df = data[democols + areacols + volcols]

    # select baseline and 2y follow up data
    baseline = df.loc[df["eventname"] == "baseline_year_1_arm_1"]
    follow = df.loc[df["eventname"] == "2_year_follow_up_y_arm_1"]

    # subjects with only baseline data
    onlyBL = [
        s for s in list(baseline["subjectkey"]) if s not in list(follow["subjectkey"])
    ]

    # subselect subjects that only have baseline data for later training
    df_onlyBL = baseline.loc[baseline["subjectkey"].isin(onlyBL)]

    # subselect subjects from baseline and follow with 2 timepoints
    df_2timepoint_bl = baseline.loc[~baseline["subjectkey"].isin(onlyBL)]
    df_2timepoint_fol = follow.loc[
        follow["subjectkey"].isin(list(df_2timepoint_bl["subjectkey"]))
    ]

    # sort dfs with two timepoints according to subjects
    df_2timepoint_bl = df_2timepoint_bl.sort_values("subjectkey").reset_index(drop=True)
    df_2timepoint_fol = df_2timepoint_fol.sort_values("subjectkey").reset_index(
        drop=True
    )

    assert (
        df_2timepoint_bl["subjectkey"].to_list()
        == df_2timepoint_fol["subjectkey"].to_list()
    ), "diff subjects!"

    # create list with demo, area, vol per baseline_only, baseline, follow
    output_list = []
    for df in [df_onlyBL, df_2timepoint_bl, df_2timepoint_fol]:
        demo = df[democols]
        area = df[["subjectkey"] + areacols]
        vol = df[["subjectkey"] + volcols]
        output_list.append([demo, area, vol])

    return output_list


def print_abcd_dict(file):
    """
    This function prints a data dictionary containing variable names and variable descriptions
    ------
    Input:
    - ABCD file as string, e.g. 'abcd_mri01' without .txt extension
    ------
    Output:
    - None
    """
    path = "../network_fusion/Package_1197473/{}.txt"
    f = path.format(file)
    header = pd.read_csv(f, header=None, sep="\t", nrows=2)
    abcd_dict = dict(zip(header.iloc[0, :], header.iloc[1, :]))
    # print key, value per line
    [print(key, ":", value) for key, value in abcd_dict.items()]


def extend_demodf(output_list, file, cols):
    """
    This function extends the ABCD demo dataframes with additional information, e.g., scannersite
    ------
    Input:
    - output_list: list of dataframes with [demo, area, vol]
    - file: ABCD file name (string), e.g., 'abcd_mri01' without .txt extension
    - cols: list of variables, which cols should be added to demodf, use print_abcd_dict() before to get
    variable description
    Note: "subjectkey", "eventname" is added as default
    ------
    Output:
    - output_list extended
    """

    # load file and add column names
    path = "../network_fusion/Package_1197473/{}.txt"
    f = path.format(file)
    header = pd.read_csv(f, header=None, sep="\t", nrows=2)
    df = pd.read_csv(f, header=None, sep="\t", skiprows=2)
    df.columns = list(header.iloc[0, :])

    # choose cols of
    cols_of_interest = ["subjectkey", "eventname"]
    cols_of_interest += cols
    df = df[cols_of_interest]

    for i in range(len(output_list)):
        # first two inputs relate to baseline data
        if i < 2:
            subdf = df.loc[df["eventname"] == "baseline_year_1_arm_1"]
            output_list[i][0] = pd.merge(output_list[i][0], subdf, on="subjectkey")
        else:
            subdf = df.loc[df["eventname"] == "2_year_follow_up_y_arm_1"]
            output_list[i][0] = pd.merge(output_list[i][0], subdf, on="subjectkey")

    return output_list


def create_data_array(dataframes, return_subjects=False):
    """
    This function creates np.arrays from dataframe and drops the subject information
    ------
    Input:
    - dataframes = list of dataframes with MRI modalities + subject identifiers
    - return_subjects = whether list of subjects should be returned, default = FALSE
    ------
    Output:
    - array_list: list of len(dataframes) containing numpy arrays (nsubjects x 68)
    - subjects: list of subjects
    """

    array_list = [
        np.array(df.select_dtypes("float64"), dtype="float64") for df in dataframes
    ]

    if return_subjects == True:
        subjects = list(dataframes[0]["subjectkey"])
        return array_list, subjects
    else:
        return array_list


def site_harmonization(array_list, site_df, site_var):
    """
    This function performs Combat harmonization per MRI modality
    Input:
    - array_list: list of arrays containing area and volume data, Note: Combat expects shape (features x subject)
    - site_df: dataframe containing demo and site information
    - site_var: variable that encodes MRI site information (here: scanner number)
    ------
    Output:
    - array_list_combat: combat harmonized MRI data with shape of input array_list, Note: transposing necessary!
    """

    array_list_combat = []

    for array in array_list:
        array_combat = neuroCombat(dat=array.T, covars=site_df, batch_col=site_var)[
            "data"
        ]
        array_list_combat.append(array_combat.T)

    return array_list_combat


def euler_exclude(mri_df, threshold=3):
    """
    This functions calculates mean euler number across hemispheres as a proxy for dataquality and
    returns a mask indicating which participants exceed mean - std * standard deviation
    Function is used for PNC data!
    -----
    Input
    - mri_df: a dataframe containing MRI subjects assuring same subjects + ordering
    - threshold: indicates {} x SD threshold, default = 3SD, decrease for stricter threshold
    ----
    Output
    - mask: boolean mask for dataframe indicating which subjects exceed threshold, pd.Series
    """

    file = "../PNC/allEuler.csv"
    df = pd.read_csv(file)

    df["avg_euler"] = (df.euler_lh + df.euler_rh) / 2
    df.sort_values(by="subject", inplace=True)

    assert list(df.subject) == list(
        mri_df.subjectkey
    ), "subjects do not match! check again"

    mask = df["avg_euler"] < df["avg_euler"].mean() - threshold * df["avg_euler"].std()

    return mask
