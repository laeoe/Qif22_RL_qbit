import numpy as np
import os

c = list()

a = np.arange(20)
b = np.ones(20)

c.append(a)
c.append(b)

c = np.array(c)
print(c)
print(np.mean(c, axis = 0))

#print('cwd', os.getcwd() + "/second_env/training_results")

#np.save(os.getcwd() + "/second_env/training_results/testfile", a)