from src.preprocess.mimic import process_mimic
from src import bootstrap

feature_cols = ["temperature", "hematocrit", "potassium", "wbc", "meanbp", "heartratehigh", "creatinine"]
time_col = "ic_t"
target_col = "antibiotics_n"
identifier_col = "identifier"
dataset = "mimic"

exp_name = "MIMIC"
input_size = len(feature_cols) + 2
context_size = len(feature_cols) + 1

hidden_dims = [16, 32, 64]
lambdas = [0.0001, 0.001, 0.01, 0.1]

# pin pandas version <2.0
df = process_mimic()
bootstrap.create_bootstap_splits(df=df, dataset_name="mimic", n_bootstrap=10, split_col=identifier_col)


for rnn_type in ["RNN", "LSTM"]:
    for run in range(10):
        print(f"Run {run}")
        bootstrap.train_bootstrap(exp_name=exp_name, dataset_name=dataset, run=run, feature_cols=feature_cols, time_col=time_col,
                                target_col=target_col, identifier_col=identifier_col, input_size=input_size, rnn_type=rnn_type,
                                context_size=context_size, hidden_dims=hidden_dims, lambdas=lambdas)

for rnn_type in ["RNN", "LSTM"]:
    for run in range(10):
        print(f"Run {run}")
        bootstrap.train_bootstrap_vanilla(exp_name=exp_name, dataset=dataset, run=run, feature_cols=feature_cols, time_col=time_col,
                                target_col=target_col, identifier_col=identifier_col, input_size=8, rnn_type=rnn_type,
                                hidden_dims=hidden_dims, action_col="antibiotics")


best_context_rnn = bootstrap.get_best_model(experiment_name=exp_name, pref="context_RNN")
results_c_rnn = bootstrap.get_test_results(dataset=dataset, exp_name=exp_name, model_name=best_context_rnn)

best_context_lstm = bootstrap.get_best_model(experiment_name=exp_name, pref="context_LSTM")
results_c_lstm = bootstrap.get_test_results(dataset=dataset, exp_name=exp_name, model_name=best_context_lstm)

best_base_rnn = bootstrap.get_best_model(experiment_name=exp_name, pref="baseline_RNN")
results_b_rnn = bootstrap.get_test_results(dataset=dataset, exp_name=exp_name, model_name=best_base_rnn)

best_base_lstm = bootstrap.get_best_model(experiment_name=exp_name, pref="baseline_LSTM")
results_b_lstm = bootstrap.get_test_results(dataset=dataset, exp_name=exp_name, model_name=best_base_lstm)

results_lr = bootstrap.get_lr_results(dataset=dataset, feature_cols=feature_cols, target_col=target_col)

results = "\n\n".join([results_c_rnn, results_c_lstm, results_b_rnn, results_b_lstm, results_lr])

with open('MIMIC_results.txt', 'w') as f:
    f.write(results)