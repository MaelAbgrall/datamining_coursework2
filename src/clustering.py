import os
import time
import sys

# libs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn.cluster as skcluster

# project files
import utils.filehandler as filehandler

# debug mode: python main.py -d
DEBUG = False
if ("-d" in sys.argv):
    DEBUG = True

# import data
dataset = filehandler.import_csv('../fer2018/fer2018.csv')
(x_train, y_train, x_validation, y_validation) = filehandler.classic_split(dataset, 0.75)

# create result folder
path = None #in case of debug
if(DEBUG == False):
    path = "result/test_" + str(time.time())
    os.makedirs(path, exist_ok=True)



# feedback
if(DEBUG == False):
    #plt.plot(val_loss, label="validation loss")
    #plt.plot(loss, label="train loss")
    #plt.title('Loss evolution')
    #plt.ylabel('Loss')
    #plt.xlabel('Iterations')
    #plt.savefig(path + "loss.png")
    pass