import numpy as np
import os
import generalized.custom_functions as custom_functions
import generalized.hyperparams as hyperparams

results_dir = hyperparams.results_dir


l = custom_functions.data_load(results_dir + "data_list")

l_ = l[-1][1][0][1]


print(l_[0])
print(len(l_))
#print(l[0][2][0])
#print(c)



#print(np.mean(c, axis = 0))

#print('cwd', os.getcwd() + "/second_env/training_results")

#np.save(os.getcwd() + "/second_env/training_results/testfile", a)