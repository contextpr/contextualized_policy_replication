from sklearn.linear_model import LogisticRegression
import numpy as np
import torch 
from typing import List, Optional, Tuple
from src.models import contextualized_sigmoid, VanillaRnn
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, auc, precision_recall_curve
import os 
from pathlib import Path
import torch.nn as nn
import json


STORAGE_PATH = Path(__file__).parent.parent / "results"

def init_run(experiment_name:str, dataset_name:str, bootstrap:Optional[int]=None) -> Path:
    """Initialize a run folder for a given dataset, experiment combination.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    dataset_name : str
        Name of the dataset.
    bootstrap : int, optional
        Id of bootstrap run, by default None

    Returns
    -------
    Path
        Path to the run folder.

    Raises
    ------
    ValueError
        If the run already exists to prevent overwriting.
    """
    
    run_path = STORAGE_PATH / dataset_name / experiment_name
    if bootstrap is not None:
        run_path = run_path / f"run_{bootstrap}"
    
    if run_path.exists():
        raise ValueError("This run already exists.")
    else:
        os.makedirs(run_path)
    return run_path 


def log_run(run_path:Path, model:torch.nn.Module, best_val:float, train_loss:List[float], val_loss:List[float]):
    """Log the results of a run.

    Parameters
    ----------
    run_path : Path
        Path to the run folder.
    model : torch.nn.Module
        The best model of the run.
    best_val : float
        The best validation loss.
    train_loss : List[float]
        List of training losses.
    val_loss : List[float]
        List of validation losses.
    """

    metrics = {"best_val": best_val, "train_loss": train_loss, "val_loss": val_loss}

    with open(run_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    torch.save(model.state_dict(), run_path / "model.pt")


def L1_bce(pred:torch.Tensor, target:torch.Tensor, theta:torch.Tensor, lamb:float, mask:torch.Tensor) -> torch.Tensor:
    """Compute the L1 loss with a L1 regularization term on the coefficients.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted probabilities.
    target : torch.Tensor
        True labels.
    theta : torch.Tensor
        Linear Coefficients.
    lamb : float
        Regularization coefficient.
    mask : torch.Tensor
        Mask to remove padded values.

    Returns
    -------
    torch.Tensor
        The loss.
    """

    zer = torch.zeros_like(theta)
    reg = nn.L1Loss(reduction="none")
    loss_fct = nn.BCELoss(reduction="none")

    mask_l1 = mask.unsqueeze(-1).expand(theta.size())

    reg_loss = lamb * reg(theta, zer) * mask_l1 
    bce = loss_fct(pred, target) * mask

    loss = bce.sum()/mask.sum() + reg_loss.sum() / mask.sum()

    return loss


def train_contextual(exp_name:str, input_size:int, context_size:int, train_loader:torch.utils.data.DataLoader, 
                     val_loader:torch.utils.data.DataLoader, lr:float=5e-4, rnn_type:str="LSTM", 
                     hidden_dims:List[int]=[16, 32, 64], lambdas:List[int]=[0.0001, 0.001, 0.01, 0.1], 
                     bootstrap:Optional[int]=None):
    """Train a contextualized model.

    Parameters
    ----------
    exp_name : string
        Name of the experiment.
    input_size: int 
        Dimension of the input.
    context_size : int 
        Dimension of the context.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader
        Validation data loader, used for early stopping.
    lr : float, optional
        Learning rate. Defaults to 5e-2.
    rnn_type : str, optional
        Recurrent architecture. Defaults to "LSTM".
    hidden_dims : List[int], optional
        List of hidden states used for GridSearch. Defaults to [16, 32, 64].
    lambdas : List[int], optional
        List of regularization values lambdas. Defaults to [0, 0.001, 0.01, 0.1].
    bootstrap : int, optional
        Number of bootstrap run. Defaults to None.
    """

    for hidden_dim in hidden_dims:
        for lamb in lambdas:
            
            model_name = f"context_{rnn_type}_{hidden_dim}_{lamb}"
            try:
                run_path = init_run(model_name, dataset_name=exp_name, bootstrap=bootstrap)
            except ValueError as e:
                print(f"Run {model_name} already exists, skipping.")
                continue
            
            model = contextualized_sigmoid(hidden_dim=hidden_dim, type=rnn_type, input_size=input_size, 
                                           context_size=context_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_prev = 1000
            best_val = 1000
            best_model = None

            train_losses = []
            val_losses = []

            no_improvement = 0

            for epoch in range(1000):
                loss_train = 0
                item = 0

                print(epoch) 
                for context, features, targets, _, mask, static in train_loader:
                    batch_size = targets.shape[0]
                    item+=1

                    optimizer.zero_grad()
                    hidden = model.init_hidden(batch_size=batch_size, static=static)

                    outs = []
                    thetas = []
                    for step in range(context.size(-1)):
                        context_step = context[:, :,step].unsqueeze(-2)
                        features_step = features[:, :,step].unsqueeze(-2)

                        out, hidden, theta = model(context=context_step, observation=features_step, hidden=hidden)
                        # we dont regularize the intercept
                        thetas.append(theta[:,:-1])
                        outs.append(out)
    
                    probs = torch.vstack(outs).T
                    thetas = torch.stack(thetas)
                    thetas = thetas.permute(1,0,2)


                    loss = L1_bce(pred=probs, target=targets, theta=thetas, lamb=lamb, mask=mask)
                    loss.backward()
                    loss_train += loss.item()

                    optimizer.step()
                
                train_losses.append(loss_train/item)
            
                with torch.no_grad():
                    v_losses = []
                    for context, features, targets, _, mask, static in val_loader:

                        batch_size = targets.shape[0] 
                        hidden = model.init_hidden(batch_size=batch_size, static=static)
                        outs = []
                        thetas = []
                        for step in range(context.size(-1)):
                            context_step = context[:, :,step].unsqueeze(-2)
                            features_step = features[:, :,step].unsqueeze(-2)
                            
                            out, hidden, theta = model(context=context_step, observation=features_step, hidden=hidden)
                            outs.append(out)
                            thetas.append(theta)

                        outs = torch.vstack(outs).T
                        thetas = torch.stack(thetas)
                        thetas = thetas.permute(1,0,2)
                        loss = L1_bce(pred=outs, target=targets, theta=thetas, lamb=0.0, mask=mask)
                        v_losses.append(loss.item())

                    val_loss = np.mean(v_losses)
                    val_losses.append(val_loss)     

                    if val_loss < best_val:
                        best_prev = best_val
                        best_val = val_loss
                        best_model = model
                        if best_prev-best_val<1e-5:
                            no_improvement += 1
                        else: 
                            no_improvement = 0
                        print(best_val)
                    
                    else:
                        no_improvement += 1
                
                if no_improvement >= 10:
                    break
                
            log_run(run_path = run_path, model=best_model, best_val=best_val, train_loss=train_losses, 
                    val_loss=val_losses)


def vanilla_bce(pred:torch.Tensor, target:torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the BCE loss for a vanilla RNN.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predictions of the model.
    target : torch.Tensor
        Target values.
    mask : torch.Tensor
        Mask to remove padded values.
    Returns
    -------
    torch.Tensor: 
        The loss
    """
    loss = nn.BCELoss(reduction="none")

    bce = loss(pred, target) * mask

    return bce.sum()/mask.sum()


def train_vanilla(exp_name:str, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
                  input_size:int, lr:float=1e-4, hidden_dims:List[int]=[16, 32, 64], rnn_type:str="LSTM", 
                  no_imp:int=10, bootstrap:Optional[bool]=None):
    """Train a baseline RNN model.

    Parameters
    ----------
    exp_name : str
        Name of the experiment.
    train_loader : torch.utils.data.Dataloader
        Dataloader for the training set.
    val_loader : torch.utils.data.Dataloader
        Dataloader for the validation set.
    input_size : int
        Size of the input.
    lr : float, optional
        Learning rate, by default 1e-4
    hidden_dims : List[int], optional
        List of hidden dimensions, train model for each value in list, by default [16, 32, 64]
    rnn_type : str, optional
        Type of RNN architecture, by default "LSTM"
    no_imp : int, optional
        Number of steps without improvement until training is stopped, by default 10
    bootstrap : Optional[bool], optional
        Bootstrap ID, by default None
    """

    for h in hidden_dims:

        model_name = f"baseline_{rnn_type}_{h}"
        try:
            run_path = init_run(model_name, dataset_name=exp_name, bootstrap=bootstrap)
        except ValueError:
            print(f"Run {model_name} already exists")
            continue

        model_v = VanillaRnn(input_size=input_size, hidden_dim=h, rnn_type=rnn_type)
        optimizer = torch.optim.Adam(model_v.parameters(), lr=lr)
        
        best_prev = 1000
        best_val = 1000
        best_model = None

        train_losses = []
        val_losses = []

        no_improvement = 0

        for epoch in range(1000):
            loss_train = 0
            item = 0

            print(epoch) 

            for features, targets, _, mask in train_loader:
                item+=1

                bs = targets.shape[0]

                optimizer.zero_grad()
                hidden = model_v.init_hidden(bs)
                outs = []
                for step in range(features.shape[-1]):
                    features_step = features[:, :,step].unsqueeze(-2)

                    out, hidden = model_v(input=features_step, hidden=hidden)
                    outs.append(out)
                probs = torch.hstack(outs)
                loss = vanilla_bce(probs, targets, mask=mask)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            
            train_losses.append(loss_train / item)

            with torch.no_grad():
                v_losses = []
                for features, targets, _, mask in val_loader:
                    bs = features.shape[0]
                    hidden = model_v.init_hidden(bs)
                    outs = []
                    for step in range(features.shape[-1]):
                        features_step = features[:, :,step].unsqueeze(-2)

                        out, hidden = model_v(input=features_step, hidden=hidden)
                        outs.append(out)
                    probs = torch.hstack(outs)
                    loss = vanilla_bce(probs, targets, mask=mask)
                    v_losses.append(loss.item())
            
                val_loss = np.mean(v_losses)
                val_losses.append(val_loss)   

                if val_loss < best_val:
                    best_prev = best_val
                    best_val = val_loss
                    best_model = model_v
                    if best_prev-best_val<1e-5:
                        no_improvement += 1
                    else: 
                        no_improvement = 0
                    print(best_val)
                
                else:
                    no_improvement += 1
                
            if no_improvement >= no_imp or best_prev-best_val<1e-5:
                break
        
        log_run(run_path = run_path, model=best_model, best_val=best_val, train_loss=train_losses, val_loss=val_losses)


def load_run(run:str, dataset_name:str, bootstrap=None)-> torch.nn.Module:
    """Load a trained model

    Parameters
    ----------
    run : str
        Name of the run
    dataset_name : str
        Name of the dataset
    bootstrap : int, optional
        ID of the bootstrap run, by default None

    Returns
    -------
    torch.nn.Module
        The trained model
    """

    p = run.split("_")
    h = int(p[2])
    rnn_type = p[1]
    run_path = STORAGE_PATH / dataset_name / run 
    if bootstrap is not None:
        run_path = run_path / f"run_{bootstrap}"

    if "adni" in dataset_name.lower() :
        context_size = 7
        input_size = 6
    
    elif "mimic" in dataset_name.lower():
        context_size = 9
        input_size = 8

    try:
        model = contextualized_sigmoid(hidden_dim=h, type=rnn_type, input_size=input_size, context_size=context_size)
        model.load_state_dict(torch.load(run_path / "model.pt"))
    except RuntimeError:
        model = VanillaRnn(input_size=input_size, hidden_dim=h, rnn_type=rnn_type)
        model.load_state_dict(torch.load(run_path / "model.pt"))

    return model


def fit_lr(x_train, y_train, x_val, y_val, Cs=[1e9, 1e8, 1e7, 1e6, 1e5, 1000, 100, 10, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
    """Fit a logistic regression model to the training data and return the model with the highest validation accuracy.

    Parameters
    ----------
    x_train : np.array
        The training features
    y_train : np.array
        The training labels
    x_val : np.array
        The validation features
    y_val : np.array
        The validation labels
    Cs : list, optional
        List of L1 penalty coefficients tested, by default [1e9, 1e8, 1e7, 1e6, 1e5, 1000, 100, 10, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        The best logistic regression model
    """

    best_acc = 0
    best_model = None

    for c in Cs:
        lr = LogisticRegression(C=c, max_iter=1000, penalty="l1", solver="liblinear", intercept_scaling=1000)
        lr.fit(x_train, y_train)
        acc = lr.score(x_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_model = lr

    return best_model


def calculate_results(pred:np.array, true:np.array)-> Tuple[float, float, float, float]:
    """Calculate the AUROC, AUPRC, Brier score and F1 score

    Parameters
    ----------
    pred : np.array
        Model predictions
    true : np.array
        True labels

    Returns
    -------
    Tuple[float, float, float, float]
        AUROC, AUPRC, Brier score and F1 score
    """
    auroc = roc_auc_score(true, pred)
    pre, rec, _ = precision_recall_curve(true, pred)
    auprc = auc(rec, pre)
    brier = brier_score_loss(true, pred)

    pred_class = pred.copy()
    pred_class[pred_class >=0.5] = 1
    pred_class[pred_class < 0.5] = 0
    f1 = f1_score(true, pred_class)

    return auroc, auprc, brier, f1
