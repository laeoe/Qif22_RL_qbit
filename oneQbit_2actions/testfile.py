import numpy as np
import os

c = list()

a = [0, 1, 2, 3, 5 ,[0, 0]]
b = [0]

c.append([a, b])

c = np.array(c)
print(c)


#print(np.mean(c, axis = 0))

#print('cwd', os.getcwd() + "/second_env/training_results")

#np.save(os.getcwd() + "/second_env/training_results/testfile", a)