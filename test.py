import pickle

import numpy as np

from gblda import GibbsLDA


with open("save.pkl", "rb") as fp:
    lda = pickle.load(fp)



