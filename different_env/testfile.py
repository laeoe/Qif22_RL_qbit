import numpy as np
import os
import custom_functions
import hyperparams

results_dir = hyperparams.results_dir


l = custom_functions.data_load(results_dir + "data_list")

l_ = l[-1][1][0][1]

while True:
    print('hi')
    break
#print(l_[0])
#print(len(l_))
#print(l[0][2][0])
#print(c)



#print(np.mean(c, axis = 0))

#print('cwd', os.getcwd() + "/second_env/training_results")

#np.save(os.getcwd() + "/second_env/training_results/testfile", a)