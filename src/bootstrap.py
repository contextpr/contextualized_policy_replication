from collections import defaultdict
from src import models, data, trainer
from src.preprocess import util as preprocessing_util
from src import util
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import os
import json
from typing import Union, List, Tuple
import torch

DATA_PATH = Path(__file__).parent.parent / "data"
STORAGE_PATH = Path(__file__).parent.parent / "results"

TRAINING_PARAMS = {"adni":{"dataset": "adni", "time_col": "t", 
                           "feature_cols": ["CDRSB_med", "CDRSB_high", "hippocampus_low", 
                                            "hippocampus_avg", "hippocampus_high"],
                           "target_col": "a_n", "identifier_col": "RID", "action_col": "a"},
                   "mimic": {"dataset": "mimimc", "time_col": "ic_t", 
                           "feature_cols": ["temperature", "hematocrit", "potassium", "wbc", "meanbp", 
                                            "heartratehigh", "creatinine"],
                           "target_col": "antibiotics_n", "identifier_col": "identifier", "action_col": "antibiotics"}}


def get_bootstrap_path(dataset:str, run:int) -> Path:
    """Get the path to the bootstrap data for a given dataset and run.

    Parameters
    ----------
    dataset : str
        The dataset to get the path for
    run : int
        The run to get the path for

    Returns
    -------
    Path
        The path to the bootstrap data
    """
    return DATA_PATH / dataset / "bootstrap" / f"run_{run}"


def get_run_path(dataset_name:str) -> Path:
    """Get the path to the results of a given experiment.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset, adni, mimic or simulation

    Returns
    -------
    Path
        The path to the results of the experiment
    """
    return STORAGE_PATH / dataset_name


def create_bootstrap_sample(df:pd.DataFrame, identifier_col:str) -> pd.DataFrame:
    """Create a bootstrap sample from a dataframe based on an ID column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to sample from
    identifier_col : str
        The column to sample on

    Returns
    -------
    pd.DataFrame
        The bootstrap sample
    """
    train_ids = df[identifier_col].unique()
    n_train_ids = len(train_ids)
    bootstrap_sample = np.random.choice(train_ids, size=n_train_ids, replace=True)

    count_dict = defaultdict(int)
    dfs = []

    for i in list(bootstrap_sample):
        
        df_sample = df[df[identifier_col] == i].copy()
        df_sample[identifier_col] = df_sample[identifier_col].astype(str)
        if count_dict[i] > 0:
            # In case an identifier is sampled twice, append a number to the end
            df_sample[identifier_col] = df_sample[identifier_col] + "_" + str(count_dict[i])
        count_dict[i] += 1
        dfs.append(df_sample)
    
    df_sample = pd.concat(dfs)
    return df_sample


def create_bootstap_splits(df:pd.DataFrame, dataset_name:str, n_bootstrap:int=10, split_col:str="identifier"):
    """Create n_bootstrap bootstrap samples and split them into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to sample from
    dataset_name : str
        The name of the dataset
    n_bootstrap : int, optional
        Number of bootstrap samples, by default 10
    split_col : str, optional
        ID column, by default "identifier"
    """
    
    np.random.seed(1234)

    for i in range(n_bootstrap):
        df_population = create_bootstrap_sample(df, identifier_col=split_col)
        train, val, test = preprocessing_util.split_df(df=df_population, split_col=split_col)
        
        path = get_bootstrap_path(dataset=dataset_name, run=i)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Bootstrap run {i} already exists, skipping.")
            continue

        train.to_csv(path / "train.csv", index=False)
        val.to_csv(path / "val.csv", index=False)
        test.to_csv(path / "test.csv", index=False)


def load_bootstrap_run(dataset:str, run:int) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a previously sampled bootstrap run.

    Parameters
    ----------
    dataset : str
        Dataset name
    run : int
        Number of the bootstrap run

    Returns
    -------
    pd.DataFrame
        The training set
    pd.DataFrame
        The validation set
    pd.DataFrame
        The test set
    """
    path = get_bootstrap_path(dataset=dataset, run=run)
    train = pd.read_csv(path / "train.csv")
    val = pd.read_csv(path / "val.csv")
    test = pd.read_csv(path / "test.csv")
    return train, val, test


def get_bootstrap_loader_context(dataset:str, run:int, feature_cols:List[str], time_col:str, target_col:str, 
                                 identifier_col:str) -> Union[DataLoader, DataLoader, DataLoader]:
    """Get CPR dataloaders for a bootstrap run.

    Parameters
    ----------
    dataset : str
        Name of the dataset
    run : int
        ID of the bootstrap run
    feature_cols : List[str]
        List of feature columns
    time_col : str
        Name of the time column
    target_col : str
        Name of the target column
    identifier_col : str
        Name of the ID column

    Returns
    -------
    DataLoader
        Training set dataloader
    DataLoader
        Validation set dataloader
    DataLoader
        Test set dataloader
    """
    train, val, test = load_bootstrap_run(dataset=dataset, run=run)
    train, val, test = preprocessing_util.standardize(train=train, val=val, test=test, feature_cols=feature_cols)

    max_length = train[time_col].max() + 1

    train.sort_values(by=[identifier_col, time_col], inplace=True)
    val.sort_values(by=[identifier_col, time_col], inplace=True)
    test.sort_values(by=[identifier_col, time_col], inplace=True)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    ds_train = data.Context_df_dataset(train, feature_cols=feature_cols, target_col=target_col,
                                        identifier_col=identifier_col, max_length=max_length)
    ds_test = data.Context_df_dataset(test, feature_cols=feature_cols, target_col=target_col, 
                                      identifier_col=identifier_col, max_length=max_length)
    ds_val = data.Context_df_dataset(val, feature_cols=feature_cols, target_col=target_col, 
                                     identifier_col=identifier_col, max_length=max_length)
    
    loader_train = DataLoader(ds_train, shuffle=True, batch_size=64)
    loader_val = DataLoader(ds_val, shuffle=True, batch_size=64)
    loader_test = DataLoader(ds_test, shuffle=False, batch_size=1)

    return loader_train, loader_val, loader_test


def get_bootstrap_loader_vanilla(dataset:pd.DataFrame, run:int, feature_cols:List[str], time_col:str, target_col:str, 
                                 identifier_col:str, action_col:str) -> Union[DataLoader, DataLoader, DataLoader]:
    """Get baseline RNN dataloaders for a bootstrap run.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataframe containing the dataset
    run : int
        ID of the bootstrap run
    feature_cols : List[str]
        List of feature columns
    time_col : str
        Name of the time column
    target_col : str
        Name of the target column
    identifier_col : str
        Name of the ID column
    action_col : str
        Name of the action column

    Returns
    -------
    DataLoader
        Training set dataloader
    DataLoader
        Validation set dataloader
    DataLoader
        Test set dataloader
    """
    train, val, test = load_bootstrap_run(dataset=dataset, run=run)
    train, val, test = preprocessing_util.standardize(train=train, val=val, test=test, feature_cols=feature_cols)

    max_length = train[time_col].max() + 1

    train.sort_values(by=[identifier_col, time_col], inplace=True)
    val.sort_values(by=[identifier_col, time_col], inplace=True)
    test.sort_values(by=[identifier_col, time_col], inplace=True)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    ds_train = data.Vanilla_df_dataset(train, feature_cols=feature_cols, target_col=target_col,
                                        identifier_col=identifier_col, max_length=max_length, action_col=action_col)
    ds_test = data.Vanilla_df_dataset(test, feature_cols=feature_cols, target_col=target_col, 
                                      identifier_col=identifier_col, max_length=max_length, action_col=action_col)
    ds_val = data.Vanilla_df_dataset(val, feature_cols=feature_cols, target_col=target_col, 
                                     identifier_col=identifier_col, max_length=max_length, action_col=action_col)
    
    loader_train = DataLoader(ds_train, shuffle=True, batch_size=64)
    loader_val = DataLoader(ds_val, shuffle=True, batch_size=64)
    loader_test = DataLoader(ds_test, shuffle=False, batch_size=1)

    return loader_train, loader_val, loader_test


def train_bootstrap(exp_name:str, dataset_name:str, run:int, feature_cols:List[str], time_col:str, target_col:str, 
                    identifier_col:str, input_size:int, context_size:int, hidden_dims:List[int], lambdas:List[float], 
                    rnn_type:str):
    """Train a CPR model for every hidden_dim x lambda combination on a bootstrap sample run.

    Parameters
    ----------
    exp_name : str
        Name of the experiment
    dataset_name : str
        Name of the dataset
    run : int
        ID of the bootstrap run
    feature_cols : List[str]
        List of feature columns x
    time_col : str
        Name of the time column (used for ordering visits)
    target_col : str
        Name of the target column
    identifier_col : str
        Name of the ID column
    input_size : int
        Size of the input
    context_size : int
        Size of the context
    hidden_dims : List[int]
        List of hidden dimensions, train a model for every hidden dimension
    lambdas : List[float]
        List of lambda values, train a model for every lambda value
    rnn_type : str
        Type of RNN to use ["RNN", "LSTM"]
    """
    loader_train, loader_val, _ = get_bootstrap_loader_context(dataset=dataset_name, run=run, 
                                                                        feature_cols=feature_cols,
                                                                        time_col=time_col,
                                                                        target_col=target_col,
                                                                        identifier_col=identifier_col)
    
    trainer.train_contextual(exp_name=exp_name, input_size=input_size, context_size=context_size, 
                             train_loader=loader_train, val_loader=loader_val, hidden_dims=hidden_dims, 
                             lambdas=lambdas, lr=5e-4, bootstrap=run, rnn_type=rnn_type)
    

def train_bootstrap_vanilla(exp_name:str, dataset:str, run:int, feature_cols:List[str], time_col:str, target_col:str, 
                            identifier_col:str, input_size:int, hidden_dims:List[int], rnn_type:str, action_col:str):
    """Train a baseline RNN model for every hidden_dim on a bootstrap sample run.

    Parameters
    ----------
    exp_name : str
        Name of the experiment
    dataset : str
        Name of the dataset
    run : int
        ID of the bootstrap run
    feature_cols : List[str]
        List of feature columns x
    time_col : str
        Name of the time column (used for ordering visits)
    target_col : str
        Name of the target column
    identifier_col : str
        Name of the ID column
    input_size : int
        Size of the input
    hidden_dims : List[int]
        List of hidden dimensions, train a model for every hidden dimension
    rnn_type : str
        Type of RNN to use ["RNN", "LSTM"]
    action_col : str
        Column containing the previous action a_{t-1}, RNN takes in [x_t, a_{t-1}}] at each timestep
    """
    loader_train, loader_val, _ = get_bootstrap_loader_vanilla(dataset=dataset, run=run, 
                                                                        feature_cols=feature_cols,
                                                                        time_col=time_col,
                                                                        target_col=target_col,
                                                                        identifier_col=identifier_col,
                                                                        action_col=action_col)
    trainer.train_vanilla(exp_name=exp_name, input_size=input_size, train_loader=loader_train,
                          val_loader=loader_val, hidden_dims=hidden_dims, lr=1e-4, rnn_type=rnn_type, bootstrap=run)


def load_model(model_name:str, experiment_name:str, bootstrap:int) -> torch.nn.Module:
    """Load a model from a given dataste and bootstrap run.

    Parameters
    ----------
    model_name : str
        Name of the model
    experiment_name : str
        Name of the experiment
    bootstrap : int
        Bootstrap run

    Returns
    -------
    torch.nn.Module
        Loaded model
    """
    model = trainer.load_run(run=model_name, dataset_name=experiment_name, bootstrap=bootstrap)
    return model


def get_best_model(experiment_name:str, pref:str, n_bootstrap=10) -> str:
    """Get the best model (validation loss) for a given prefix and dataset. Eg use prefix=context_RNN and dataset_name=mimic
    to get the best CPR model using an RNN encoder on the MIMIC dataset.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    pref : str
        Prefix of the model class to get the best model for
    n_bootstrap : int, optional
        Number of bootstrap runs, by default 10

    Returns
    -------
    str
        Name of the best model
    """
    p = get_run_path(experiment_name)
    model_configs = [run for run in os.listdir(p) if run.startswith(pref)]
    results = {}

    for model_config in model_configs:
        val_losses = []
        for boostrap_run in range(n_bootstrap):
             run_result = json.load(open(p / model_config / f"run_{boostrap_run}" / "metrics.json", "r"))
             val_losses.append(run_result["best_val"])
        results[model_config] = np.array(val_losses).mean()
            
    return min(results, key=results.get)


def get_test_results(dataset:str, exp_name:str, model_name:str):
    """_summary_

    Parameters
    ----------
    dataset : str
        _description_
    exp_name : str
        _description_
    model_name : str
        _description_
    """
    p = get_run_path(exp_name) / model_name

    aurocs, auprcs, briers, f1s = [], [], [], []
    for boostrap_run in range(10):

        model = load_model(model_name=model_name, experiment_name=exp_name, bootstrap=boostrap_run)
        run_params = TRAINING_PARAMS[dataset.lower()]

        if not "baseline" in model_name:
            _,_, loader_test = get_bootstrap_loader_context(dataset=dataset, run=boostrap_run, 
                                                            feature_cols=run_params["feature_cols"], 
                                                            time_col=run_params["time_col"], 
                                                            target_col=run_params["target_col"], 
                                                            identifier_col=run_params["identifier_col"])

            pred_context, true_context = models.model_predict(model, loader_test)
        else:
            _,_, loader_test = get_bootstrap_loader_vanilla(dataset=dataset, run=boostrap_run, 
                                                            feature_cols=run_params["feature_cols"], 
                                                            time_col=run_params["time_col"], 
                                                            target_col=run_params["target_col"],
                                                            identifier_col=run_params["identifier_col"], 
                                                            action_col=run_params["action_col"])

            pred_context, true_context = models.vanilla_predict(model, loader_test)

        auroc, auprc, brier, f1 = trainer.calculate_results(pred = pred_context, true=true_context)
        aurocs.append(auroc)
        auprcs.append(auprc)
        briers.append(brier)
        f1s.append(f1)

    final_str = util.format_results(model_name=model_name, dataset=dataset, auprcs=auprcs, aurocs=aurocs, briers=briers, f1s=f1s)
    return final_str



def bootstrap_lr(run:str, dataset:str, feature_cols:List[str], target_col:str)-> Tuple[float, float, float, float]:
    """_summary_

    Parameters
    ----------
    run : str
        _description_
    dataset : str
        _description_
    feature_cols : List[str]
        _description_
    target_col : str
        _description_

    Returns
    -------
    Tuple[float, float, float, float]
        _description_
    """
    train, val, test = load_bootstrap_run(dataset=dataset, run=run)

    if dataset.lower() == "mimic":
        train, val, test = preprocessing_util.standardize(train=train, val=val, test=test, feature_cols=feature_cols)

    x_train = train[feature_cols].values
    y_train = train[target_col].values

    x_val = val[feature_cols].values
    y_val = val[target_col].values

    x_test = test[feature_cols].values
    y_test = test[target_col].values

    lr = trainer.fit_lr(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    y_pred_test = lr.predict_proba(x_test)[:,1]
    auroc, auprc, brier, f1 = trainer.calculate_results(pred = y_pred_test, true=y_test)

    return auroc, auprc, brier, f1


def get_lr_results(dataset:str, feature_cols:List[str], target_col:str) -> str:
    """Get results for logistic regression

    Parameters
    ----------
    dataset : str
        Dataset name
    feature_cols : List[str]
        List of feature columns
    target_col : str
        Target column

    Returns
    -------
    str
        Formated string containing results
    """
    aurocs, auprcs, briers, f1s = [], [], [], []

    for i in range(10):
        auroc, auprc, brier, f1 = bootstrap_lr(dataset=dataset, feature_cols=feature_cols, target_col=target_col, run=i)
        aurocs.append(auroc)
        auprcs.append(auprc)
        briers.append(brier)
        f1s.append(f1)

    return util.format_results(model_name="Logistic_Regression", dataset=dataset, aurocs=aurocs, auprcs=auprcs, briers=briers,
                               f1s=f1s)