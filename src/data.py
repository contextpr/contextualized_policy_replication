from torch.utils.data import Dataset
import torch
import pandas as pd

import torch.nn.functional as nn 
from typing import List, Optional


class Context_df_dataset(Dataset):
    def __init__(self, df:pd.DataFrame, max_length:int, feature_cols:List[str], static_cols:Optional[List[str]]=None, 
                 target_col:str="a_n", identifier_col:str="RID", add_intercept_yn:bool=True):
        """Dataset class for CPR models

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the preprocessed data
        max_length : int
            Maximum length of a sequence
        feature_cols : List[str]
            List of feature columns
        static_cols : Optional[List[str]], optional
            List of static columns, by default None
        target_col : str, optional
            Name of the target column, by default "a_n"
        identifier_col : str, optional
            Name of the identifier column identifying a sequence, by default "RID"
        add_intercept_yn : bool, optional
            Add an intercept to the observation-to-action mapping, by default True
        """
        self.df = df
        self.patients = self.df[identifier_col].unique()
        self.context_cols = feature_cols 
        self.context_cols = self.context_cols + [target_col]
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.identifier_col = identifier_col
        self.max_length = max_length
        self.add_intercept_yn = add_intercept_yn
        self.static_cols = static_cols
    
    def add_sequence_start_col(data:torch.Tensor) -> torch.Tensor:
        """Add a column of zeros to the context for the fist visit

        Parameters
        ----------
        data : torch.Tensor
            The context data

        Returns
        -------
        torch.Tensor
            The context data with a column of zeros prepended
        """
        rnn_start_col = torch.zeros((data.shape[0],1))
        data_additional_col = torch.cat((rnn_start_col, data), 1)

        return data_additional_col

    def add_dummy_row(data:torch.Tensor) -> torch.Tensor:
        """Add a dummy row to the context. Set to all zeros except for the first column which is set to 1 to indicate
        the start of the context.

        Parameters
        ----------
        data : torch.Tensor
            The context data

        Returns
        -------
        torch.Tensor
            The context data with a dummy row prepended
        """
        start_row = torch.zeros((1, data.shape[1]))
        start_row[0][0] = 1
        new_data = torch.cat((start_row, data), 0)

        return new_data

    def add_intercept(data:torch.Tensor) -> torch.Tensor:
        """Add an intercept to x_t for the observation-to-action mapping

        Parameters
        ----------
        data : torch.Tensor
            x_t

        Returns
        -------
        torch.Tensor
            x_t with an intercept
        """
        interc = torch.ones((data.shape[0],1))
        data_interc = torch.cat((data, interc), 1)

        return data_interc

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient = self.patients[index]
        data = self.df[self.df[self.identifier_col] == patient]
        target = torch.tensor(data[self.target_col].values, dtype=torch.float)

        features = torch.tensor(data[self.feature_cols].values, dtype=torch.float)
        context = torch.tensor(data[self.context_cols].values, dtype=torch.float) 
        context = Context_df_dataset.add_sequence_start_col(context)

        context = Context_df_dataset.add_dummy_row(context)
        # Context contains values up to t-1, so remove the last row
        context = context[:-1,:]

        if self.add_intercept_yn:
            features_interc = Context_df_dataset.add_intercept(features)
        else:
            features_interc = features

        seq_length = target.shape[0]
        # Add mask to indicate which values are padded
        mask = torch.cat([torch.ones(seq_length), torch.zeros(self.max_length-seq_length)])
        static = torch.zeros((1))

        if self.static_cols is not None:
            static = data[self.static_cols].iloc[0].values
            static = torch.tensor(static, dtype=torch.float)

        return (nn.pad(context.T, (0,self.max_length-seq_length)), 
                nn.pad(features_interc.T, (0, self.max_length-seq_length)), 
                nn.pad(target, (0, self.max_length - seq_length)), 
                patient,
                mask,
                static
        )


class Vanilla_df_dataset(Dataset):
    def __init__(self, df, max_length, feature_cols, static_cols=None, target_col="a_n", action_col="a", identifier_col="RID"):
        self.df = df
        self.patients = df[identifier_col].unique()
        self.feature_cols = feature_cols.copy()
        self.target_col = target_col
        self.identifier_col = identifier_col
        self.feature_cols += [action_col]
        self.max_length = max_length
        self.static_cols = static_cols

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):
        patient = self.patients[index]
        data = self.df[self.df[self.identifier_col] == patient]

        target = torch.tensor(data[self.target_col].values, dtype=torch.float)

        features = torch.tensor(data[self.feature_cols].values, dtype=torch.float)
    
        seq_length = target.shape[0]
        mask = torch.cat([torch.ones(seq_length), torch.zeros(self.max_length-seq_length)])

        if self.static_cols is not None:
            static = data[self.static_cols].iloc[0].values
            static = torch.tensor(static, dtype=torch.float)
    
        return (nn.pad(features.T, (0, self.max_length-seq_length)), 
                nn.pad(target, (0, self.max_length - seq_length)), 
                patient,
                mask
        )