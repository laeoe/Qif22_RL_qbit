import numpy as np
import os



a = np.arange(200)

#print('cwd', os.getcwd() + "/second_env/training_results")

np.save(os.getcwd() + "/second_env/training_results/testfile", a)