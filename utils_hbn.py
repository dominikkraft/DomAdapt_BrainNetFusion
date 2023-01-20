"""
Utility functions for the Healthy Brain Network (HBN). 
Author: Dominik Kraft
"""


import pandas as pd


def load_loris_data():
    """
    Function that loads demographic, site and diagnostic information
    based on LORIS database download.
    Note: we downloaded data from LORIS + COINS and merged both to use
    all potentially existing data.
    """
    # load individual dataframes for demo, mri site and diagnostics
    demo = pd.read_csv(
        "../hbn/nonMRI/HBN_demo_neu1708.csv",
        sep=",",
        skiprows=[1],
        usecols=["Identifiers", "Basic_Demos,Sex", "Basic_Demos,Age"],
    )

    mri = pd.read_csv(
        "../hbn/nonMRI/HBN_scannerinfo_neu1708.csv",
        sep=",",
        usecols=["MRI_Track,Scan_Location", "Identifiers"],
        skiprows=[1],
    )

    diag_cols_loris = [
        "Diagnosis_ClinicianConsensus,DX_01",
        "Diagnosis_ClinicianConsensus,DX_02",
        "Diagnosis_ClinicianConsensus,DX_03",
        "Diagnosis_ClinicianConsensus,DX_04",
        "Diagnosis_ClinicianConsensus,DX_05",
        "Diagnosis_ClinicianConsensus,DX_06",
        "Diagnosis_ClinicianConsensus,DX_07",
        "Diagnosis_ClinicianConsensus,DX_08",
        "Diagnosis_ClinicianConsensus,DX_09",
        "Diagnosis_ClinicianConsensus,DX_10",
    ]

    diagnosis = pd.read_csv(
        "../hbn/nonMRI/HBN_diagnosis_neu1708.csv",
        sep=",",
        usecols=diag_cols_loris + ["Identifiers"],
        skiprows=[1],
    )

    # merge dataframe
    df = pd.merge(demo, mri, on="Identifiers").merge(diagnosis, on="Identifiers")

    # * perform some data wrangling
    # rename subjects
    df["Identifiers"] = [
        "sub-" + x for x in df["Identifiers"].str.replace(",assessment", "")
    ]
    # rename columns
    df.rename(
        columns={
            "Basic_Demos,Sex": "Sex",
            "Basic_Demos,Age": "Age",
            "MRI_Track,Scan_Location": "Scan_Location",
        },
        inplace=True,
    )

    # rename diagnosis column to match with coins
    diagnosis_dict = {
        "Diagnosis_ClinicianConsensus,DX_01": "DX_01",
        "Diagnosis_ClinicianConsensus,DX_02": "DX_02",
        "Diagnosis_ClinicianConsensus,DX_03": "DX_03",
        "Diagnosis_ClinicianConsensus,DX_04": "DX_04",
        "Diagnosis_ClinicianConsensus,DX_05": "DX_05",
        "Diagnosis_ClinicianConsensus,DX_06": "DX_06",
        "Diagnosis_ClinicianConsensus,DX_07": "DX_07",
        "Diagnosis_ClinicianConsensus,DX_08": "DX_08",
        "Diagnosis_ClinicianConsensus,DX_09": "DX_09",
        "Diagnosis_ClinicianConsensus,DX_10": "DX_10",
    }

    df.rename(mapper=diagnosis_dict, axis="columns", inplace=True)

    # drop duplicates
    df.drop_duplicates(subset=["Identifiers"], inplace=True)
    return df


def load_coins_data():
    """
    Function that loads demographic, site and diagnostic information
    based on COINS database download.
    """
    # load individual dataframes for demo, mri site and diagnostics
    demo = pd.read_csv(
        "../hbn/nonMRI/coins/assessment_data/9994_Basic_Demos_20220818.csv",
        sep=",",
        skiprows=[1],
        usecols=["EID", "Sex", "Age"],
    )

    mri = pd.read_csv(
        "../hbn/nonMRI/coins/assessment_data/9994_MRI_Track_20220818.csv",
        sep=",",
        usecols=["Scan_Location", "EID"],
        skiprows=[1],
    )

    diag_cols_coins = [
        "DX_01",
        "DX_02",
        "DX_03",
        "DX_04",
        "DX_05",
        "DX_06",
        "DX_07",
        "DX_08",
        "DX_09",
        "DX_10",
    ]

    diagnosis = pd.read_csv(
        "../hbn/nonMRI/coins/assessment_data/9994_ConsensusDx_20220818.csv",
        sep=",",
        usecols=diag_cols_coins + ["EID"],
        skiprows=[1],
    )

    # merge dataframe
    df = pd.merge(demo, mri, on="EID").merge(diagnosis, on="EID")

    # * perform some data wrangling
    # rename columns
    df.rename(columns={"EID": "Identifiers"}, inplace=True)

    # rename subjects
    df["Identifiers"] = [
        "sub-" + x for x in df["Identifiers"].str.replace(",assessment", "")
    ]
    # drop duplicates

    df.drop_duplicates(subset=["Identifiers"], inplace=True)

    return df
