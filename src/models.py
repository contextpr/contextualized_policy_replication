from torch import nn
import torch

from typing import Tuple
import numpy as np

class contextualized_sigmoid(nn.Module):
    def __init__(self, hidden_dim=16, n_layers=1, context_size=6, input_size=6, type="RNN"):
        super(contextualized_sigmoid, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.context_size = context_size
        self.input_size = input_size
        self.type = type

        if self.type == "RNN":
            self.rnn = nn.RNN(self.context_size, self.hidden_dim, self.n_layers, batch_first=True)
        elif self.type=="LSTM":
            self.rnn = nn.LSTM(self.context_size, self.hidden_dim, self.n_layers, batch_first=True)
        else:
            raise ValueError
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.input_size)
        self.relu = nn.ReLU()
    
    def init_hidden(self, batch_size, static=None):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        if self.type == "LSTM":
            hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(self.n_layers, batch_size, self.hidden_dim))

        if self.type == "LSTM":
            hidden, cell = hidden

        # Encode static features in initial hidden state
        if static is not None:
            hidden = hidden[:,:, :self.hidden_dim-static.shape[-1]]
            static = torch.unsqueeze(static, 0)
            hidden = torch.cat([static, hidden], dim=-1)

        if self.type == "LSTM":
            hidden = (hidden, cell)

        return hidden
    
    def get_theta(self, context, hidden):
        theta, hidden = self.rnn(context, hidden)
        theta = theta.contiguous().view(-1, self.hidden_dim)
        theta = self.relu(theta)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc(theta)
        return theta, hidden

    def forward(self, context, observation, hidden=None):

        batch_size = context.size(0)
        sig = nn.Sigmoid()

        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size)

        theta, hidden = self.get_theta(context, hidden)

        if self.context_size != 1:
            logits_mat = observation.squeeze() @ theta.T
        
        else:
            logits_mat =  (observation @ theta.T).squeeze()
        
        if len(logits_mat.shape) == 1:
            logits_mat = logits_mat.unsqueeze(0)
        
        logits = torch.diagonal(logits_mat)
        prob = sig(logits)
        
        return prob, hidden, theta
    

def model_predict(model, loader, evaluate_from=0) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probability of taking an action for each sequence in a dataset usiing a CPR model.

    Parameters
    ----------
    model : torch.nn.Module
        The CPR model
    loader : torch.utils.data.DataLoader
        The dataset to predict on
    evaluate_from : int, optional
        Timestep to start evaluating from, by default 0

    Returns
    -------
    np.ndarray
        The predicted probabilities
    np.ndarray
        The true labels
    """
    preds = []
    target = []

    for context, features, targets, _, mask, static in loader:
        bs = targets.shape[0]
        outs = []
        hidden = model.init_hidden(bs, static)
        seq_len = int(mask.sum().item()) 

        for step in range(seq_len):
            context_step = context[:,:,step].unsqueeze(-2)
            features_step = features[:,:,step].unsqueeze(-2)
            out, hidden, _ = model(context=context_step, observation=features_step, hidden=hidden)
            if step >= evaluate_from:
                outs.append(out)
        
        targets = targets.T[evaluate_from:seq_len,:]
        probs = torch.vstack(outs)

        preds.append(probs)
        target.append(targets)

    pred_np = torch.concat(preds).detach().numpy().squeeze()
    true_np = torch.concat(target).numpy()

    return pred_np, true_np


class VanillaRnn(nn.Module):
    def __init__(self, input_size=6, hidden_dim=64, n_layers=1, rnn_type="RNN"):
        super(VanillaRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_size = input_size
        self.typ = rnn_type
        
        if self.typ == "RNN":
            self.rnn = nn.RNN(self.input_size, self.hidden_dim, self.n_layers, batch_first=True)
        elif self.typ == "LSTM":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim, self.n_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        if self.typ == "LSTM":
            hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden

    def forward(self, input, hidden=None):
        batch_size = input.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size)
        
        out,hidden = self.rnn(input, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.sig(out)

        return out, hidden


def vanilla_predict(model, loader, evaluate_from=0) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probability of taking an action for each sequence in a dataset usiing a vanilla model.

    Parameters
    ----------
    model : torch.nn.Module
        The CPR model
    loader : torch.utils.data.DataLoader
        The dataset to predict on
    evaluate_from : int, optional
        Timestep to start evaluating from, by default 0

    Returns
    -------
    np.ndarray
        The predicted probabilities
    np.ndarray
        The true labels
    """
    preds = []
    target = []        
    for features, targets, _, mask in loader:

        bs = targets.shape[0]

        outs = []
        hidden = model.init_hidden(bs)
        seq_len = int(mask.sum().item()) 
        hidden = model.init_hidden(bs) 
        outs = []
        for step in range(seq_len):
            features_step = features[:,:,step].unsqueeze(-2)
            out, hidden = model(input=features_step, hidden=hidden)
            if step >= evaluate_from:
                outs.append(out)
        
        targets = targets.T[evaluate_from:seq_len,:] 
        target.append(targets)
        probs = torch.vstack(outs)
        preds.append(probs)

    pred_np = torch.concat(preds).detach().numpy().squeeze()
    true_np = torch.concat(target).numpy()

    return pred_np, true_np
