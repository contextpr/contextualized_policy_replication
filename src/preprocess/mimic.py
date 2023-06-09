import psycopg2 as psql
from pathlib import Path
import pandas.io.sql as sqlio
from datetime import datetime
from src.preprocess.sql import *
import numpy as np

import pandas as pd

import os 

from pathlib import Path

#Preprocessing adapted from 
#https://github.com/vanderschaarlab/clairvoyance/tree/db0c1502bef5e938f789120ef0d3e05db907ea06/datasets/mimic_data_extraction

SQL_PATH = Path(__file__).parent / "sql"
MIMIC_FEATS = ["temperature", "hematocrit", "hemoglobin", "potassium", "wbc", "meanbp", "heartratehigh", "creatinine"]

def connect():
    """Connect to the MIMIC database.

    Returns
    -------
    psycopg2.connection
        The connection to the MIMIC database
    """
    try:
        conn = psql.connect(dbname="mimic", user="postgres", password="postgres", host="localhost",)
        conn.cursor().execute("SET search_path TO mimiciii")
        return conn
    except (Exception, psql.DatabaseError) as error:
        print(error)


def create_aux_tables(conn):
    """Create the auxiliary tables for the MIMIC database.

    Parameters
    ----------
    conn : psycopg2.connection
        The connection to the MIMIC database
    """
    with conn:
        with conn.cursor() as cursor:
            query = "".join(open(SQL_PATH / "cohort.sql").readlines())
            cursor.execute(query)
            query = "".join(open(SQL_PATH / "easychart.sql").readlines())
            cursor.execute(query)
            query = "".join(open(SQL_PATH / "easylabs.sql").readlines())
            cursor.execute(query)
            query = "".join(open(SQL_PATH / "easyvent.sql").readlines())
            cursor.execute(query)


def get_longitudinal_features(conn)->pd.DataFrame:
    """Extract longitudinal features from database connection conn.

    Parameters
    ----------
    conn : psql.connection
        The connection to the MIMIC database containing the auxiliary tables

    Returns
    -------
    pd.DataFrame
        Dataframe containing the longitudinal features
    """
    longitudinal = sqlio.read_sql_query(query_longitudinal, conn)
    longitudinal = fill_missing_days(longitudinal)
    longitudinal["subject_id"] = longitudinal["subject_id"].astype("int64")
    longitudinal["hadm_id"] = longitudinal["hadm_id"].astype("int64")
    longitudinal["icustay_id"] = longitudinal["icustay_id"].astype("int64")
    longitudinal["ic_t"] = longitudinal["ic_t"].astype("int64")
    return longitudinal


def get_demographic_data(conn) -> pd.DataFrame:
    """Extract static demographic data from database connection conn.

    Parameters
    ----------
    conn : psql.connection
        The connection to the MIMIC database containing the auxiliary tables

    Returns
    -------
    pd.DataFrame
        Dataframe containing the demographic data
    """
    return sqlio.read_sql_query("SELECT subject_id, hadm_id, icustay_id, gender, first_admit_age FROM cohort", conn)


def get_antibiotics_data(conn) -> pd.DataFrame:
    """Extract antibiotic data from database connection conn. Taken from 
    https://github.com/vanderschaarlab/clairvoyance/tree/db0c1502bef5e938f789120ef0d3e05db907ea06/datasets/mimic_data_extraction

    Parameters
    ----------
    conn : psql.connection
        The connection to the MIMIC database containing the auxiliary tables

    Returns
    -------
    pd.DataFrame
        Dataframe containing the antibiotic data
    """

    abx_iv = sqlio.read_sql_query(query_antibiotics + "select * from ab_tbl", conn)
    abx_iv_all = abx_iv[~abx_iv["antibiotic_name"].isnull()]

    patients_antibiotics = abx_iv_all
    patients_antibiotics["intime"] = patients_antibiotics["intime"].apply(
        lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
    )
    patients_antibiotics["antibiotic_time"] = patients_antibiotics["antibiotic_time"].apply(
        lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
    )
    patients_antibiotics["ic_t"] = patients_antibiotics["antibiotic_time"] - patients_antibiotics["intime"]
    patients_antibiotics["ic_t"] = patients_antibiotics["ic_t"].apply(lambda x: (x.days))

    def g(x):
        antibiotics = set(x["antibiotic_name"])
        x = x.head(n=1)
        x["antibiotic_name"].iloc[0] = antibiotics

        return x

    patients_antibiotics = patients_antibiotics.groupby(["icustay_id", "ic_t"]).apply(g).reset_index(drop=True)
    patients_antibiotics = patients_antibiotics.drop(columns=["intime", "antibiotic_time"])
    patients_antibiotics = patients_antibiotics.rename(columns={"antibiotic_name": "antibiotics"})

    return patients_antibiotics


def fill_missing_days(longitudinal: pd.DataFrame) -> pd.DataFrame:
    """Fill missing days in the longitudinal data. Eg add row 2 if only day 0, 1, 3, 4 are present.

    Parameters
    ----------
    longitudinal : pd.DataFrame
        Dataframe containing the longitudinal data

    Returns
    -------
    pd.DataFrame
        Dataframe containing the longitudinal data with missing days filled in
    """
    def f(x):
        days = set(x["ic_t"])
        l = len(x)
        missing = set(np.arange(l, dtype=float)) - days
        hadm_id = x["hadm_id"].iloc[0]
        icustay_id = x["icustay_id"].iloc[0]
        subject_id = x["subject_id"].iloc[0]
        for d in missing:
            x = x.append(
                {"hadm_id": hadm_id, "icustay_id": icustay_id, "subject_id": subject_id, "ic_t": d}, ignore_index=True
            )
        return x

    return longitudinal.groupby(["icustay_id"]).apply(f).reset_index(drop=True)


def process_mimic():
    """Process MIMIC-III dataset.

    Returns:
        pd.DataFrame: Processed dataset.
    """

    con = connect()
    longitudinal = get_longitudinal_features(con)
    antibiotics = get_antibiotics_data(con)
    demographic = get_demographic_data(con)

    # Filter out patients with no longitudinal features or no antibiotics features
    icustay_ids = set(antibiotics["icustay_id"]) & set(longitudinal["icustay_id"])
    longitudinal = longitudinal[longitudinal["icustay_id"].isin(icustay_ids)]
    antibiotics = antibiotics[antibiotics["icustay_id"].isin(icustay_ids)]

    # Merge datasets
    merged_dataset = pd.merge(
        longitudinal,
        antibiotics,
        how="left",
        left_on=["icustay_id", "hadm_id", "ic_t"],
        right_on=["icustay_id", "hadm_id", "ic_t"],
    )

    merged_dataset = pd.merge(
        merged_dataset,
        demographic,
        how="left",
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "hadm_id"],
    )

    df = merged_dataset[["subject_id", "hadm_id", "ic_t", "antibiotics", "charttime", "gender", "first_admit_age"] 
                        + MIMIC_FEATS]

    # Filter out patients with missing features
    a = df[df[MIMIC_FEATS].isna().sum(axis=1)==0]
    a = a.sort_values(["subject_id", "hadm_id", "ic_t"])

    def numerate(df):
        df = df.sort_values("ic_t")
        ref = list(range(len(df)))

        df["ref"] = ref
        if (df["ref"] == df["ic_t"]).all():
            return df.drop(columns=["ref"])
    
    # 6 visits starting from t=0
    a = a[a["ic_t"]<=5]
    b = a.groupby(["subject_id", "hadm_id"]).apply(numerate).drop(columns=["subject_id", "hadm_id"]).reset_index().drop(columns=["level_2"])
    b["antibiotics"] = b["antibiotics"].notna() * 1.0
    b["subject_id"] = b["subject_id"].astype(int)

    b = b.sort_values(["subject_id", "hadm_id", "ic_t"])
    b["antibiotics_n"] = b.groupby(["subject_id", "hadm_id"])["antibiotics"].shift(-1)
    b = b[b["antibiotics_n"].notna()]

    b["identifier"] = b["subject_id"].astype(str) + b["hadm_id"].astype(str)
    b.rename(columns={"first_admit_age": "age"}, inplace=True)

    return b 