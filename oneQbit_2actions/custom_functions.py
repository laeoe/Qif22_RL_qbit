import pickle
import os

folder_name = "/oneQbit_2actions/"
cwd = os.getcwd()
results_dir = cwd + folder_name + "training_results/"


def data_save(lists, filename):
    """Takes list of lists and saves it into filename"""
    outfile = open(filename, 'wb')
    pickle.dump(lists, outfile)
    outfile.close()
    
def data_load(filename):
    infile = open(filename, 'rb')
    lists = pickle.load(infile)
    infile.close()
    return lists