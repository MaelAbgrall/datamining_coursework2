import os
import time
import sys

# libs
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import view.plot as viewer

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

# defining hyperparameters
Kcluster = 7 # there is 7 emotions

# create result folder
path = None #in case of debug
if(DEBUG == False):
    path = "result/test_" + str(time.time())
    os.makedirs(path, exist_ok=True)

print("start fit")
kmean = skcluster.KMeans(n_clusters=Kcluster)
start = time.time()
kmean.fit(x_train)
print("fitting done in ", time.time() - start, "s")
y_predict = kmean.predict(x_train)

viewer.accuracy_plots("7", y_predict, y_train)

# feedback
if(DEBUG == False):
    #plt.plot(val_loss, label="validation loss")
    #plt.plot(loss, label="train loss")
    #plt.title('Loss evolution')
    #plt.ylabel('Loss')
    #plt.xlabel('Iterations')
    #plt.savefig(path + "loss.png")
    pass