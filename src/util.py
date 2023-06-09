import numpy as np

def format_results(model_name:str, dataset:str, aurocs:np.array, auprcs:np.array, briers:np.array, f1s:np.array) -> str:
    final_str = f"Dataset: {dataset} | Model: {model_name}\n"
    final_str += f"========================\n"
    final_str += f"AUROC:\t {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f} \n"
    final_str += f"AUPRC:\t {np.mean(auprcs):.3f} +/- {np.std(auprcs):.3f} \n"
    final_str += f"Brier:\t {np.mean(briers):.3f} +/- {np.std(briers):.3f} \n"
    final_str += f"F1:\t {np.mean(f1s):.3f} +/- {np.std(f1s):.3f} \n"
    return final_str
