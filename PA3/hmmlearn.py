# use this file to learn hidden markov model
# Expected: generate hmmmodel.txt

""" Import necessary library"""
from __future__ import division
import time
import math
import copy
import pickle
import sys
from io import open

model_file="hmmmodel.txt"


""" Utility function"""




if __name__=="__main__":
    train_file = sys.argv[1]
    print(train_file)
    open(model_file, "w")
