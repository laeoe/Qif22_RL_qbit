

import numpy as np
import qutip
import matplotlib
from matplotlib import pyplot, animation
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
from colour import Color

folder_name = "/oneQbit_2actions/"
cwd = os.getcwd()
results_dir = cwd + folder_name + "training_results/"

initialState = np.array([np.sqrt((5+np.sqrt(5))/10),np.sqrt(2/(5+np.sqrt(5)))])
targetState = np.array([-(1+np.sqrt(5))/(np.sqrt(2*(5+np.sqrt(5)))), np.sqrt(2.0/(5+np.sqrt(5)))])

#def data_save(lists, filename):
#    """Takes list of lists and saves it into filename"""
#    outfile = open(filename, 'wb')
#    pickle.dump(lists, outfile)
#    outfile.close()
    
def data_load(filename):
    infile = open(filename, 'rb')
    lists = pickle.load(infile)
    infile.close()
    return lists

l = data_load(results_dir + "data_list")

pointList = []

for wavefunction in l[-1][1][0][1]:
    pointList.append([wavefunction[0]+1j*wavefunction[1],wavefunction[2]+1j*wavefunction[3]])

print(l[-1][1][0][1])

s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])
smats = [s1,s2,s3]

vecListX = []
vecListY = []
vecListZ = []

for dummyvector in pointList[:]:
    vecListX.append([np.matmul(np.conj(dummyvector),np.matmul(s1,dummyvector)).real])
    vecListY.append([np.matmul(np.conj(dummyvector),np.matmul(s2,dummyvector)).real])
    vecListZ.append([np.matmul(np.conj(dummyvector),np.matmul(s3,dummyvector)).real])

initialStateVec = [np.matmul(np.conj(initialState),np.matmul(paulimat,initialState)).real for paulimat in smats]
targetStateVec = [np.matmul(np.conj(targetState),np.matmul(paulimat,targetState)).real for paulimat in smats]



fig = pyplot.figure()
ax = Axes3D(fig, azim=-40, elev=30)
sphere = qutip.Bloch(axes=ax)

red = Color("red")
colors = list((red.range_to(Color("green"),len(vecListX))))
colorStrings = []

for c in colors:
    colorStrings.append(str(c))
    
def animate(i):
    sphere.clear()
    sphere.add_vectors(initialStateVec)
    sphere.add_vectors(targetStateVec)
    sphere.point_color = [colorStrings[:i+1]]
    sphere.add_points([vecListX[:i+1],vecListY[:i+1],vecListZ[:i+1]])
    sphere.make_sphere()
    return ax

def init():
    sphere.vector_color = ['r','g']
    return ax

ani = animation.FuncAnimation(fig, animate, np.arange(len(vecListX)),init_func=init, blit=False, repeat=False)
ani.save('bloch_sphere.gif', fps=5)

