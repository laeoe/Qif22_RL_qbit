import numpy as np
import os
import custom_functions
import hyperparams

results_dir = hyperparams.results_dir


l = custom_functions.data_load(results_dir + "data_list")

print("shape", np.shape(l[0][2][0]))
print(l[0][2][0][0])
#print(c)



#print(np.mean(c, axis = 0))

#print('cwd', os.getcwd() + "/second_env/training_results")

#np.save(os.getcwd() + "/second_env/training_results/testfile", a)