import os
import time
import sys

# project files
import utils.filehandler as filehandler

# debug mode: python main.py -d
DEBUG = False
if ("-d" in sys.argv):
    DEBUG = True


# import data
#filehandler.get_data("mypath", sep=False)

# create result folder
if(DEBUG == False):
    path = "result/test_" + str(time.time())
    os.makedirs(path, exist_ok=True)