import numpy as np
from numpy import save,load
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['font.size'] = 17
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from scipy.interpolate import interp1d
import scipy.sparse as sps
from scipy.sparse import linalg as sps_linalg
import scipy.linalg as scipy_linalg
from importlib import reload
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import PCA
from abc import ABC,abstractmethod
import time
import queue
import sys
import os
from os.path import join,exists
codefolder = "/home/jf4241/SHORT"
os.chdir(codefolder)

# A class to specify a boundary value problem. 

class BVP(ABC):
    def __init__(self):
        super().__init__()
        return

