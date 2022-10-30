import os

folder_name = "/oneQbit_2actions/"
cwd = os.getcwd()
results_dir = cwd + folder_name + "training_results/"

#Agent Hyperparams
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_steps = 20000
PRINT_INTERVAL = update_interval * 100

depth_firt_layer = 4

