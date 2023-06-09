from pathlib import Path
import pandas as pd

from src.preprocess.util import encode

RAW_PATH = Path(__file__).parent.parent.parent / "data" / "adni" / "raw"

def load_adni(filename="ADNIMERGE.csv"):
    """Load ADNI data from csv file

    Parameters
    ----------
    filename : str, optional
        Name of the csv file, by default "ADNIMERGE.csv"

    Returns
    -------
    pd.DataFrame
        Dataframe containing the ADNI data
    """
    
    tab = pd.read_csv(RAW_PATH / f"{filename}.csv")
    return tab


def process_adni():
    """Process ADNI data. 
    Code adapted from https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/interpole/adni/data.py

    Returns
    -------
    pd.DataFrame
        Dataframe containing the processed ADNI data
    """


    df = load_adni("adnimerge")
    df = df[~df.CDRSB.isna()]
    df = df[~df.DX.isna()]

    state = ['CN', 'MCI', 'Dementia']
    state += ['{} to {}'.format(s0, s1) for s0 in state for s1 in state if s0 != s1]
    state_dict = {s:i for s,i in zip(state,range(len(state)))}

    visc = ["bl", "m06"] + ["m{}".format(k*6) for k in range(2,20)]

    # only select subsequent visits
    rids = [df[df.VISCODE == vis].RID.unique() for vis in visc]
    for i in range(1, len(rids)):
        rids[i] = [rid for rid in rids[i] if rid in rids[i-1]]
    
    df = df[df.VISCODE.isin(visc)]
    for vis, rid in zip(visc, rids):
        df = df[df.RID.isin(rid) | (df.VISCODE != vis)]
    
    df_processed = []

    for rid in rids[0]:

        traj = dict()
        traj["a_raw"] = []
        traj["z0"] = []
        traj["z1"] = []
        traj["t"] = []
        traj["age"] = []
        traj["gender"] = []

        t = 0

        df1 = df[df.RID == rid]
        for vis in visc:

            df2 = df1[df1.VISCODE == vis]
            if df2.empty:
                break

            a = 0 if df2.Hippocampus.isna().values[0] else 1
            z0 = "low" if df2.CDRSB.values[0] == 0 else "med" if df2.CDRSB.values[0] <= 2.5 else "high"
            z1 = "no" if df2.Hippocampus.isna().values[0] else "low" if df2.Hippocampus.values[0] < 6642-.5*1225 else "avg" if df2.Hippocampus.values[0] <= 6642+.5*1225 else "high"
            age = df2.AGE.values[0]
            gender = df2.PTGENDER.values[0]

            traj["a_raw"].append(a)
            traj["z0"].append(z0)
            traj["z1"].append(z1)
            traj["t"].append(t)

            traj["age"].append(age)
            traj["gender"].append(gender)

            t += 1
        
        traj["l"] = t

        # Only look at patients with two or more visits
        if traj["l"]>1:
            
            traj["z0"] = traj["z0"][:-1]
            traj["z1"] = traj["z1"][:-1]
            traj["t"] = traj["t"][:-1]
            traj["a"] = traj["a_raw"][:-1]

            # Predict action from previous visit features
            traj["a_n"] = traj["a_raw"][1:]
            df_obs = pd.DataFrame.from_dict({"a": traj["a"], "CDRSB": traj["z0"], "hippocampus": traj["z1"], 
            "a_n":traj["a_n"], "t": traj["t"]})

            # add time independent features
            df_obs["RID"] = rid
            df_obs["age"] = traj["age"][0]
            df_obs["gender"] = traj["gender"][0]
            df_processed.append(df_obs)
    
    adni_processed = pd.concat(df_processed).reset_index(drop=True)
    return adni_processed


def encode_adni(df):
    """Encode CDRSB and hippocampus columns

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the preprocessed ADNI data

    Returns
    -------
    pd.DataFrame
        Dataframe containing the encoded ADNI data
    """
    df = encode(df=df, col="CDRSB", remove_col="low")
    df = encode(df=df, col="hippocampus", remove_col="no")

    return df
