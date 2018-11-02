import os
import time
import sys

# libs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# project files
import utils.filehandler as filehandler

# debug mode: python main.py -d
DEBUG = False
if ("-d" in sys.argv):
    DEBUG = True

# import data
dataset = filehandler.get_data("mypath", sep=False)

# create result folder
path = None  # in case of debug
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